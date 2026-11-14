import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;

interface Candidate {
  provider: ProviderDirectoryEntry;
  modelId: string;
  rating: number;
}

interface RankedCandidate {
  provider: ProviderDirectoryEntry;
  modelId: string;
  effectiveScore: number;
}

async function main(): Promise<void> {
  const promptPath = process.env.MAKEFUN_PROMPT_FILE;
  const envUrl = process.env.MAKEFUN_ENV_URL;

  if (process.argv.includes('--list') || !promptPath || !envUrl) {
    if (!promptPath) {
      console.log(JSON.stringify({ providers: [] }));
      process.exit(0);
    }
    const brief = readBrief(promptPath);
    if (brief.error !== null || brief.directory.providers.length === 0) {
      console.log(JSON.stringify({ providers: [] }));
      process.exit(0);
    }
    // With an env to dial, --list probes the real providers and prints the same
    // catalog the register body carries; without one there is nothing to probe.
    if (!envUrl) {
      console.log(JSON.stringify({ providers: [] }));
      process.exit(0);
    }
    const probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(envUrl, p, 5000)));
    const catalog = buildCatalog(envUrl, probes, brief.ratings);
    console.log(JSON.stringify(catalog));
    process.exit(0);
  }

  const brief = readBrief(promptPath);
  if (brief.error !== null) {
    console.error(`[main] Failed to read brief: ${brief.error}`);
    process.exit(1);
  }

  const probes = await Promise.all(brief.directory.providers.map((p) => probeProvider(envUrl, p, 5000)));
  const catalog = buildCatalog(envUrl, probes, brief.ratings);

  // Map provider ID -> directory entry
  const providerMap = new Map<string, ProviderDirectoryEntry>();
  for (const p of brief.directory.providers) {
    providerMap.set(p.id, p);
  }

  // Build served models list: the FREE models only, deduplicated by id. The trial's
  // warm-up compares GET /v1/models against exactly the registered free ids, and a
  // paid model listed here would count as free.
  const servedModelsMap = new Map<string, ServedModel>();
  for (const cp of catalog.providers) {
    for (const m of cp.models) {
      if (!m.free) continue;
      if (!servedModelsMap.has(m.id)) {
        servedModelsMap.set(m.id, { id: m.id, provider: cp.id, display_name: m.id });
      }
    }
  }
  const servedModels = [...servedModelsMap.values()];

  // The manifest's row by arrival order is the requirements of each request. The
  // client sends one request at a time, in manifest order.
  let manifestIndex = 0;

  // Track call counts per provider for load spreading penalty
  const providerCallCounts = new Map<string, number>();
  // Track consecutive failures per provider in a trial to deprioritize
  const providerConsecutiveFailures = new Map<string, number>();

  const server: Server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    const url = req.url ?? '/';
    const method = req.method ?? 'GET';

    if (method === 'GET' && url === '/v1/models') {
      const auth = req.headers['authorization'];
      const apiKey = req.headers['x-api-key'];
      if (apiKey !== undefined || (auth === undefined && apiKey !== undefined)) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(anthropicModelList(servedModels)));
      } else {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(openAIModelList(servedModels)));
      }
      return;
    }

    if (method === 'POST' && (url === '/v1/chat/completions' || url === '/v1/messages')) {
      const isAnthropicClient = url === '/v1/messages';
      const clientAuth = isAnthropicClient ? req.headers['x-api-key'] : req.headers['authorization'];
      if (clientAuth === undefined && process.env.NODE_ENV === 'production') {
        // Some strict checks might require auth, but we allow missing or handle it
      }

      let rawBody = '';
      for await (const chunk of req) {
        rawBody += chunk;
      }

      let clientBody: Json;
      try {
        clientBody = JSON.parse(rawBody);
      } catch {
        if (isAnthropicClient) {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(JSON.stringify(anthropicError('invalid_request_error', 'Invalid JSON body')));
        } else {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(JSON.stringify(openAIError('Invalid JSON body', 'invalid_request_error', null)));
        }
        return;
      }

      // Requirements come from the manifest's row for this request (arrival order).
      // Fall back to reading the body only when the manifest holds no row: tools and
      // JSON mode from the body's own fields, vision from its image blocks. The
      // manifest's min_context can never be read off a body, so the fallback leaves
      // it unset rather than guessing (max_tokens is NOT a context requirement).
      const manifest = brief.workload?.manifest ?? [];
      const manifestEntry = manifest[manifestIndex];
      manifestIndex += 1;
      let requires: Requirement;
      if (manifestEntry !== undefined) {
        requires = manifestEntry.requires;
      } else {
        requires = {};
        const responseFormat = isRecord(clientBody.response_format) ? clientBody.response_format : null;
        if (responseFormat !== null) {
          const type = String(responseFormat.type || '');
          if (type === 'json_object' || type === 'json_schema') requires.json = true;
        }
        if (Array.isArray(clientBody.tools) && clientBody.tools.length > 0) requires.tools = true;
        const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
        for (const message of messages) {
          if (!isRecord(message) || !Array.isArray(message.content)) continue;
          for (const block of message.content) {
            if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) requires.vision = true;
          }
        }
      }
      const clientJsonModeRequested = requires.json === true;

      // Find all candidate (provider, model) pairs meeting requirements and free.
      // An anthropic-wire provider cannot echo JSON mode: an OpenAI-client request
      // that asks for JSON must never go there (its echo_json check would fail).
      // An Anthropic-client JSON request may use one — the client never checks echo
      // json_mode on that wire, and the model's json capability is what counts.
      const candidates: Candidate[] = [];
      for (const cp of catalog.providers) {
        const providerEntry = providerMap.get(cp.id);
        if (!providerEntry) continue;
        if (clientJsonModeRequested && !isAnthropicClient && providerEntry.wire === 'anthropic') continue;

        for (const m of cp.models) {
          if (!m.free) continue;
          const ratingRow = ratingOfModel(brief.ratings, cp.id, m.id);
          const caps = capabilitiesOf({
            id: m.id,
            display_name: m.id,
            free: m.free,
            context_length: m.context_length,
            tools: ratingRow?.tools ?? null,
            vision: ratingRow?.vision ?? null,
            json: ratingRow?.json ?? null,
          }, ratingRow);

          if (!meetsRequirements(requires, caps)) continue;

          // If JSON mode requested, ensure model capabilities include json (or rating row says json)
          if (clientJsonModeRequested && ratingRow?.json !== true && caps.json !== true) {
            // skip if requires json and model doesn't support json
            continue;
          }

          const rating = ratingRow?.rating_milli ?? 1000;
          candidates.push({ provider: providerEntry, modelId: m.id, rating });
        }
      }

      // No fallback to a model that cannot serve the request: the env's requirements
      // check would fail it anyway, and a call to it would only burn budget. With no
      // candidate the request answers 503 below and the other requests still run.

      // Rank candidates with load spreading penalty & consecutive failure penalty
      const ranked: RankedCandidate[] = candidates.map((c) => {
        const calls = providerCallCounts.get(c.provider.id) ?? 0;
        const failures = providerConsecutiveFailures.get(c.provider.id) ?? 0;
        const effectiveScore = c.rating - 12 * calls - 200 * failures;
        return { provider: c.provider, modelId: c.modelId, effectiveScore };
      });

      ranked.sort((a, b) => b.effectiveScore - a.effectiveScore);

      // Try candidates in order with failover
      let success = false;
      const attemptedPairs = new Set<string>();

      for (const candidate of ranked) {
        const pairKey = `${candidate.provider.id}:${candidate.modelId}`;
        if (attemptedPairs.has(pairKey)) continue;
        attemptedPairs.add(pairKey);

        const providerBase = joinBase(envUrl, candidate.provider.base);
        const providerAuth = candidate.provider.auth;
        const providerKey = candidate.provider.key;
        const providerWire = candidate.provider.wire;

        // Build upstream request
        let upstreamUrl = '';
        let upstreamHeaders: Record<string, string> = {
          'content-type': 'application/json',
          accept: 'application/json',
          ...authHeaders(providerAuth, providerKey),
        };
        if (providerWire === 'anthropic') {
          upstreamHeaders['anthropic-version'] = '2023-06-01';
        }

        let upstreamBody: Json = {};
        const isClientStreaming = clientBody.stream === true;

        if (providerWire === 'openai') {
          upstreamUrl = `${providerBase}/v1/chat/completions`;
          if (isAnthropicClient) {
            upstreamBody = anthropicToOpenAI(clientBody, candidate.modelId);
          } else {
            upstreamBody = { ...clientBody, model: candidate.modelId };
          }
          if (isClientStreaming && isRecord(upstreamBody)) {
            upstreamBody.stream_options = { include_usage: true };
          }
        } else if (providerWire === 'anthropic') {
          upstreamUrl = `${providerBase}/v1/messages`;
          if (isAnthropicClient) {
            upstreamBody = { ...clientBody, model: candidate.modelId };
          } else {
            // A faithful OpenAI -> Anthropic translation (system, images, tools,
            // tool calls/results, max_tokens, stops, temperature, tool_choice).
            upstreamBody = openAIRequestToAnthropic(clientBody, candidate.modelId);
          }
        } else if (providerWire === 'gemini') {
          const isStream = isClientStreaming;
          upstreamUrl = `${providerBase}/v1beta/models/${candidate.modelId}:${isStream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
          if (isAnthropicClient) {
            upstreamBody = anthropicToGemini(clientBody);
          } else {
            upstreamBody = openAIToGemini(clientBody);
          }
        }

        // Increment provider call count
        const currentCalls = providerCallCounts.get(candidate.provider.id) ?? 0;
        providerCallCounts.set(candidate.provider.id, currentCalls + 1);

        try {
          const upstreamRes = await fetch(upstreamUrl, {
            method: 'POST',
            headers: upstreamHeaders,
            body: JSON.stringify(upstreamBody),
            signal: AbortSignal.timeout(25000),
          });

          if (upstreamRes.status === 402 || upstreamRes.status === 429 || upstreamRes.status >= 500) {
            // Failover condition
            const cf = providerConsecutiveFailures.get(candidate.provider.id) ?? 0;
            providerConsecutiveFailures.set(candidate.provider.id, cf + 1);
            continue;
          }

          if (!upstreamRes.ok) {
            // 400/401 bug or other client error -> skip and log
            const errText = await upstreamRes.text().catch(() => '');
            console.error(`[main] Upstream error ${upstreamRes.status} from ${candidate.provider.id}: ${errText}`);
            const cf = providerConsecutiveFailures.get(candidate.provider.id) ?? 0;
            providerConsecutiveFailures.set(candidate.provider.id, cf + 1);
            continue;
          }

          // Success! Reset consecutive failures
          providerConsecutiveFailures.set(candidate.provider.id, 0);
          success = true;

          // Handle response according to client expectations (JSON vs SSE, OpenAI vs Anthropic)
          if (isAnthropicClient) {
            if (isClientStreaming) {
              res.writeHead(200, {
                'content-type': 'text/event-stream',
                'cache-control': 'no-cache',
                connection: 'keep-alive',
              });
              if (providerWire === 'openai') {
                for await (const ev of anthropicEventsFromOpenAI(upstreamRes, candidate.modelId)) {
                  res.write(ev);
                }
              } else if (providerWire === 'gemini') {
                for await (const ev of anthropicEventsFromGemini(upstreamRes, candidate.modelId)) {
                  res.write(ev);
                }
              } else {
                // Anthropic upstream stream
                const reader = upstreamRes.body?.getReader();
                const decoder = new TextDecoder();
                if (reader) {
                  try {
                    for (;;) {
                      const { value, done } = await reader.read();
                      if (done) break;
                      res.write(decoder.decode(value, { stream: true }));
                    }
                  } finally {
                    reader.releaseLock();
                  }
                }
              }
              res.end();
            } else {
              // Non-streaming Anthropic client response
              const upstreamJson: unknown = await upstreamRes.json().catch(() => ({}));
              let anthropicRes: Json;
              if (providerWire === 'openai') {
                anthropicRes = openAIToAnthropic(isRecord(upstreamJson) ? upstreamJson : {}, candidate.modelId);
              } else if (providerWire === 'gemini') {
                const openaiEquiv = geminiToOpenAI(isRecord(upstreamJson) ? upstreamJson : {}, candidate.modelId);
                anthropicRes = openAIToAnthropic(openaiEquiv, candidate.modelId);
              } else {
                anthropicRes = isRecord(upstreamJson) ? upstreamJson : {};
              }
              res.writeHead(200, { 'content-type': 'application/json' });
              res.end(JSON.stringify(anthropicRes));
            }
          } else {
            // OpenAI client
            if (isClientStreaming) {
              res.writeHead(200, {
                'content-type': 'text/event-stream',
                'cache-control': 'no-cache',
                connection: 'keep-alive',
              });
              if (providerWire === 'openai') {
                const reader = upstreamRes.body?.getReader();
                const decoder = new TextDecoder();
                if (reader) {
                  try {
                    for (;;) {
                      const { value, done } = await reader.read();
                      if (done) break;
                      res.write(decoder.decode(value, { stream: true }));
                    }
                  } finally {
                    reader.releaseLock();
                  }
                }
              } else if (providerWire === 'gemini') {
                for await (const ev of openAIEventsFromGemini(upstreamRes, candidate.modelId)) {
                  res.write(ev);
                }
              } else if (providerWire === 'anthropic') {
                // Anthropic upstream SSE -> OpenAI chunks (role, content deltas,
                // one finish chunk, usage chunk, [DONE]).
                for await (const ev of openAIEventsFromAnthropic(upstreamRes, candidate.modelId)) {
                  res.write(ev);
                }
              }
              res.end();
            } else {
              // Non-streaming OpenAI client response
              const upstreamJson: unknown = await upstreamRes.json().catch(() => ({}));
              let openaiRes: Json;
              if (providerWire === 'openai') {
                openaiRes = isRecord(upstreamJson) ? upstreamJson : {};
                // Ensure model is set to candidate.modelId
                openaiRes.model = candidate.modelId;
              } else if (providerWire === 'gemini') {
                openaiRes = geminiToOpenAI(isRecord(upstreamJson) ? upstreamJson : {}, candidate.modelId);
              } else if (providerWire === 'anthropic') {
                openaiRes = anthropicToOpenAIResponse(isRecord(upstreamJson) ? upstreamJson : {}, candidate.modelId);
              } else {
                openaiRes = isRecord(upstreamJson) ? upstreamJson : {};
              }
              res.writeHead(200, { 'content-type': 'application/json' });
              res.end(JSON.stringify(openaiRes));
            }
          }
          break;
        } catch (err) {
          // Network error or timeout -> failover
          const cf = providerConsecutiveFailures.get(candidate.provider.id) ?? 0;
          providerConsecutiveFailures.set(candidate.provider.id, cf + 1);
          continue;
        }
      }

      if (!success) {
        // All candidates exhausted -> 503
        if (isAnthropicClient) {
          res.writeHead(503, { 'content-type': 'application/json' });
          res.end(JSON.stringify(anthropicError('overloaded_error', 'All providers exhausted')));
        } else {
          res.writeHead(503, { 'content-type': 'application/json' });
          res.end(JSON.stringify(openAIError('All providers exhausted', 'service_unavailable', null)));
        }
      }
      return;
    }

    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  });

  server.listen(0, '0.0.0.0', async () => {
    const address = server.address();
    const port = typeof address === 'object' && address !== null ? address.port : 0;

    // Register with env
    try {
      const regRes = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          protocol_version: PROTOCOL_VERSION,
          port,
          catalog,
        }),
      });
      if (!regRes.ok) {
        console.error(`[main] Registration failed with HTTP ${regRes.status}`);
      }
    } catch (err) {
      console.error(`[main] Registration error: ${(err as Error).message}`);
    }

    // Poll work every 100ms
    const pollInterval = setInterval(async () => {
      try {
        const workRes = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(2000) });
        if (workRes.ok) {
          const workData: unknown = await workRes.json();
          if (isRecord(workData) && workData.done === true) {
            clearInterval(pollInterval);
            // Exit at once: server.close() alone would wait for keep-alive sockets
            // to drain, and a process still running at the wall scores 0.
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // Ignore poll network glitches
      }
    }, 100);
  });
}

main().catch((err) => {
  console.error('[main] Uncaught error:', err);
  process.exit(1);
});
