import { isRecord, stringOf, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      return {};
    }
  }
  return {};
}

/** Translate an Anthropic Messages request body into an OpenAI Chat Completions request body. */
export function anthropicToOpenAI(body: Json, modelId: string): Json {
  const messages: Json[] = [];
  if (typeof body.system === 'string' && body.system.trim() !== '') {
    messages.push({ role: 'system', content: body.system });
  } else if (Array.isArray(body.system)) {
    const sysText = body.system
      .map((b) => (isRecord(b) && typeof b.text === 'string' ? b.text : ''))
      .filter((s) => s !== '')
      .join('\n');
    if (sysText !== '') messages.push({ role: 'system', content: sysText });
  }

  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of rawMessages) {
    if (!isRecord(msg)) continue;
    const role = typeof msg.role === 'string' ? msg.role : 'user';
    const content = msg.content;

    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }

    if (Array.isArray(content)) {
      const parts: Json[] = [];
      const toolCalls: Json[] = [];
      const toolResults: { toolCallId: string; content: string }[] = [];

      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ type: 'text', text: block.text });
        } else if (block.type === 'image') {
          const p = partOf(block);
          if (p !== null) parts.push(p);
        } else if (block.type === 'tool_use') {
          const id = typeof block.id === 'string' ? block.id : `call_${toolCalls.length}`;
          const name = typeof block.name === 'string' ? block.name : '';
          const args = isRecord(block.input) ? JSON.stringify(block.input) : '{}';
          toolCalls.push({ id, type: 'function', function: { name, arguments: args } });
        } else if (block.type === 'tool_result') {
          const toolCallId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
          const resText = toolResultText(block.content);
          if (toolCallId !== '') toolResults.push({ toolCallId, content: resText });
        }
      }

      if (toolResults.length > 0) {
        for (const tr of toolResults) {
          messages.push({ role: 'tool', tool_call_id: tr.toolCallId, content: tr.content });
        }
        // A user turn can carry text beside its tool results: keep the text as its own user message.
        if (parts.length > 0) {
          const hasImage = parts.some((p) => p.type === 'image_url');
          messages.push({ role: 'user', content: hasImage ? parts : parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n') });
        }
      } else if (toolCalls.length > 0) {
        const msgObj: Json = { role, content: parts.length > 0 ? parts : null, tool_calls: toolCalls };
        messages.push(msgObj);
      } else if (parts.length > 0) {
        const hasImage = parts.some((p) => p.type === 'image_url');
        messages.push({ role, content: hasImage ? parts : parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n') });
      }
    }
  }

  const out: Json = { model: modelId, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;

  if (Array.isArray(body.stop_sequences)) {
    out.stop = body.stop_sequences;
  } else if (typeof body.stop_sequences === 'string') {
    out.stop = [body.stop_sequences];
  }

  if (Array.isArray(body.tools)) {
    out.tools = body.tools.map((t) => {
      if (!isRecord(t)) return t;
      return {
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.input_schema,
        },
      };
    });
  }

  if (isRecord(body.tool_choice)) {
    const tc = body.tool_choice;
    if (tc.type === 'auto') out.tool_choice = 'auto';
    else if (tc.type === 'any') out.tool_choice = 'required';
    else if (tc.type === 'tool' && typeof tc.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: tc.name } };
    }
  }

  if (typeof body.stream === 'boolean') {
    out.stream = body.stream;
    if (body.stream) {
      out.stream_options = { include_usage: true };
    }
  }

  return out;
}

/**
 * Translate an OpenAI Chat Completions request body into an Anthropic Messages
 * request body. Mirror image of anthropicToOpenAI: system messages become the
 * system prompt, image_url data URLs become base64 image sources, tool definitions
 * become Anthropic tools with input_schema, assistant tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, and max_tokens / stop /
 * temperature / top_p / tool_choice pass through. NOTE: an Anthropic-wire provider
 * cannot echo JSON mode, so callers must not route a JSON-mode OpenAI request here.
 */
export function openAIRequestToAnthropic(body: Json, modelId: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (assistant && Array.isArray(raw.tool_calls) && raw.tool_calls.length > 0) {
        const blocks: Json[] = [];
        if (content !== '') blocks.push({ type: 'text', text: content });
        for (const tc of raw.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof tc.id === 'string' && tc.id !== '' ? tc.id : `call_${blocks.length}`,
            name: typeof tc.function.name === 'string' ? tc.function.name : '',
            input: argumentsOf(tc.function.arguments),
          });
        }
        messages.push({ role: 'assistant', content: blocks });
      } else {
        messages.push({ role: assistant ? 'assistant' : 'user', content });
      }
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        if (url.startsWith('data:')) {
          const comma = url.indexOf(',');
          if (comma !== -1) {
            const meta = url.slice(5, comma);
            blocks.push({ type: 'image', source: { type: 'base64', media_type: meta.split(';')[0] || 'image/png', data: url.slice(comma + 1) } });
          }
        } else if (url !== '') blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (assistant && Array.isArray(raw.tool_calls)) {
      for (const tc of raw.tool_calls) {
        if (!isRecord(tc) || !isRecord(tc.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof tc.id === 'string' && tc.id !== '' ? tc.id : `call_${blocks.length}`,
          name: typeof tc.function.name === 'string' ? tc.function.name : '',
          input: argumentsOf(tc.function.arguments),
        });
      }
    }
    if (blocks.length > 0) messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
  }

  const out: Json = { model: modelId, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop.map(String);
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An Anthropic Messages SSE stream as OpenAI Chat Completions chunks: the role
 * chunk, text deltas, tool_call deltas (input_json_delta accumulates into one
 * function.arguments string), exactly one finish chunk, a final usage chunk, then
 * [DONE]. The provider's message_start carries input_tokens and its message_delta
 * the stop reason and output_tokens.
 */
export async function* openAIEventsFromAnthropic(response: Response, modelId: string): AsyncGenerator<string> {
  const bodyStream = response.body;
  if (bodyStream === null) return;
  const reader = bodyStream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  let roleSent = false;
  let finished = false;
  const toolArgs = new Map<number, string>();
  const toolNames = new Map<number, string>();
  const toolIds = new Map<number, string>();

  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model: modelId, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const stopOf = (reason: string): string => {
    if (reason === 'max_tokens') return 'length';
    if (reason === 'tool_use') return 'tool_calls';
    if (reason === 'refusal') return 'content_filter';
    return 'stop';
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let eventName: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (dataLines.length === 0) return out;
    let data: unknown;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      if (typeof data.message.usage.input_tokens === 'number') inputTokens = data.message.usage.input_tokens;
      return out;
    }
    if (eventName === 'content_block_start' && isRecord(data.content_block)) {
      if (!roleSent) {
        roleSent = true;
        out.push(chunk({ role: 'assistant', content: '' }, null));
      }
      if (data.content_block.type === 'tool_use') {
        const index = typeof data.index === 'number' ? data.index : 0;
        toolNames.set(index, typeof data.content_block.name === 'string' ? data.content_block.name : '');
        toolIds.set(index, typeof data.content_block.id === 'string' ? data.content_block.id : `call_${index}`);
        toolArgs.set(index, '');
      }
      return out;
    }
    if (eventName === 'content_block_delta' && isRecord(data.delta)) {
      if (!roleSent) {
        roleSent = true;
        out.push(chunk({ role: 'assistant', content: '' }, null));
      }
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        out.push(chunk({ content: data.delta.text }, null));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        toolArgs.set(index, (toolArgs.get(index) ?? '') + data.delta.partial_json);
      }
      return out;
    }
    if (eventName === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = stopOf(data.delta.stop_reason);
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
      return out;
    }
    if (eventName === 'message_stop') {
      if (!roleSent) {
        roleSent = true;
        out.push(chunk({ role: 'assistant', content: '' }, null));
      }
      const calls: Json[] = [...toolArgs.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([index, args]) => ({ index, id: toolIds.get(index) ?? `call_${index}`, type: 'function', function: { name: toolNames.get(index) ?? '', arguments: args } }));
      if (calls.length > 0) out.push(chunk({ tool_calls: calls }, null));
      finished = true;
      out.push(chunk({}, pendingStop ?? 'stop'));
      out.push(
        `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model: modelId, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`,
      );
      out.push('data: [DONE]\n\n');
      return out;
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    if (!roleSent) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, pendingStop ?? 'stop');
    yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model: modelId, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
    yield 'data: [DONE]\n\n';
  }
}

/** Translate an Anthropic Messages response into an OpenAI Chat Completions response. */
export function anthropicToOpenAIResponse(body: Json, modelId: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' && block.id !== '' ? block.id : `call_${toolCalls.length}`,
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const stopRaw = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  let finishReason = 'stop';
  if (stopRaw === 'max_tokens') finishReason = 'length';
  else if (stopRaw === 'tool_use') finishReason = 'tool_calls';
  else if (stopRaw === 'refusal') finishReason = 'content_filter';
  if (toolCalls.length > 0) finishReason = 'tool_calls';
  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const outputTokens = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  };
}

/** Translate an OpenAI Chat Completions response into an Anthropic Messages response. */
export function openAIToAnthropic(body: Json, modelId: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  const message = choice !== null && isRecord(choice.message) ? choice.message : {};
  const contentBlocks: Json[] = [];

  const text = typeof message.content === 'string' ? message.content : '';
  if (text !== '') contentBlocks.push({ type: 'text', text });

  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const tc of toolCalls) {
    if (!isRecord(tc)) continue;
    const fn = isRecord(tc.function) ? tc.function : {};
    const name = typeof fn.name === 'string' ? fn.name : '';
    const args = argumentsOf(fn.arguments);
    const id = typeof tc.id === 'string' && tc.id !== '' ? tc.id : `toolu_${toolCalls.indexOf(tc)}`;
    contentBlocks.push({ type: 'tool_use', id, name, input: args });
  }

  if (contentBlocks.length === 0) contentBlocks.push({ type: 'text', text: '' });

  const finishReason = choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
  let stopReason = 'end_turn';
  if (finishReason === 'length') stopReason = 'max_tokens';
  else if (finishReason === 'tool_calls') stopReason = 'tool_use';
  else if (finishReason === 'content_filter') stopReason = 'refusal';

  const usage = isRecord(body.usage) ? body.usage : {};
  const inputTokens = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0;
  const outputTokens = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0;

  return {
    id: typeof body.id === 'string' && body.id !== '' ? body.id : 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    content: contentBlocks,
    model: typeof body.model === 'string' ? body.model : modelId,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

/** Translate OpenAI SSE frames into Anthropic SSE frames. */
export async function* anthropicEventsFromOpenAI(response: Response, modelId: string): AsyncGenerator<string> {
  const bodyStream = response.body;
  if (bodyStream === null) return;
  const reader = bodyStream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let inputTokens = 0;
  let outputTokens = 0;
  let textIndex = -1;
  const toolIndexes = new Map<number, number>();
  let nextIndex = 0;
  let pendingStop: string | null = null;
  let started = false;
  let finished = false;

  const sseEvent = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          content: [],
          model: modelId,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };

  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const stopReasonOf = (reason: string): string => {
    if (reason === 'length') return 'max_tokens';
    if (reason === 'tool_calls') return 'tool_use';
    if (reason === 'content_filter') return 'refusal';
    return 'end_turn';
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        if (typeof chunk.usage.prompt_tokens === 'number') inputTokens = chunk.usage.prompt_tokens;
        if (typeof chunk.usage.completion_tokens === 'number') outputTokens = chunk.usage.completion_tokens;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());

      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }

      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }

      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(String(choice.finish_reason));
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }

  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** Translate an OpenAI Chat Completions request into a Gemini generateContent request body. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemInstructionParts: Json[] = [];

  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of messages) {
    if (!isRecord(msg)) continue;
    const role = typeof msg.role === 'string' ? msg.role : 'user';
    const content = msg.content;

    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemInstructionParts.push({ text });
      continue;
    }

    const geminiRole = role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];

    if (typeof content === 'string' && content !== '') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url)) {
          const url = typeof part.image_url.url === 'string' ? part.image_url.url : '';
          if (url.startsWith('data:')) {
            const comma = url.indexOf(',');
            if (comma !== -1) {
              const meta = url.slice(5, comma);
              const data = url.slice(comma + 1);
              const mimeType = meta.split(';')[0] || 'image/png';
              parts.push({ inlineData: { mimeType, data } });
            }
          } else if (url !== '') {
            parts.push({ fileData: { fileUri: url, mimeType: 'image/png' } });
          }
        }
      }
    }

    if (Array.isArray(msg.tool_calls)) {
      for (const tc of msg.tool_calls) {
        if (!isRecord(tc)) continue;
        const fn = isRecord(tc.function) ? tc.function : {};
        const name = typeof fn.name === 'string' ? fn.name : '';
        const args = argumentsOf(fn.arguments);
        parts.push({ functionCall: { name, args } });
      }
    }

    if (role === 'tool') {
      const toolCallId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : 'call';
      const responseText = typeof content === 'string' ? content : JSON.stringify(content);
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: toolCallId,
              response: { output: responseText },
            },
          },
        ],
      });
      continue;
    }

    if (parts.length > 0) {
      contents.push({ role: geminiRole, parts });
    }
  }

  const geminiBody: Json = { contents };
  if (systemInstructionParts.length > 0) {
    geminiBody.systemInstruction = { parts: systemInstructionParts };
  }

  const generationConfig: Json = {};
  if (typeof body.max_tokens === 'number') generationConfig.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') generationConfig.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Array.isArray(body.stop)) {
    generationConfig.stopSequences = body.stop.map(String);
  } else if (typeof body.stop === 'string') {
    generationConfig.stopSequences = [body.stop];
  }

  const responseFormat = isRecord(body.response_format) ? body.response_format : null;
  if (responseFormat !== null) {
    const type = stringOf(responseFormat.type);
    if (type === 'json_object' || type === 'json_schema') {
      generationConfig.responseMimeType = 'application/json';
    }
  }

  if (Object.keys(generationConfig).length > 0) {
    geminiBody.generationConfig = generationConfig;
  }

  if (Array.isArray(body.tools)) {
    const functionDeclarations = body.tools.map((t) => {
      if (!isRecord(t)) return t;
      const fn = isRecord(t.function) ? t.function : {};
      return {
        name: typeof fn.name === 'string' ? fn.name : '',
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : undefined,
      };
    });
    geminiBody.tools = [{ functionDeclarations }];
  }

  return geminiBody;
}

/** Translate an Anthropic Messages request into a Gemini generateContent request body. */
export function anthropicToGemini(body: Json): Json {
  const openAIEquivalent = anthropicToOpenAI(body, 'model');
  return openAIToGemini(openAIEquivalent);
}

/** Translate a Gemini generateContent response into an OpenAI Chat Completions response. */
export function geminiToOpenAI(body: Json, modelId: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  const content = candidate !== null && isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];

  let text = '';
  const toolCalls: Json[] = [];

  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') {
      text += part.text;
    } else if (isRecord(part.functionCall)) {
      const fc = part.functionCall;
      const name = typeof fc.name === 'string' ? fc.name : '';
      const args = isRecord(fc.args) ? fc.args : {};
      toolCalls.push({
        id: `call_${name}_${toolCalls.length}`,
        type: 'function',
        function: { name, arguments: JSON.stringify(args) },
      });
    }
  }

  const finishReasonRaw = candidate !== null ? stringOf(candidate.finishReason) : 'STOP';
  let finishReason = 'stop';
  if (finishReasonRaw === 'MAX_TOKENS') finishReason = 'length';
  else if (finishReasonRaw === 'SAFETY' || finishReasonRaw === 'RECITATION') finishReason = 'content_filter';
  if (toolCalls.length > 0) finishReason = 'tool_calls';

  const usageMeta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const promptTokens = typeof usageMeta.promptTokenCount === 'number' ? usageMeta.promptTokenCount : 0;
  const completionTokens = typeof usageMeta.candidatesTokenCount === 'number' ? usageMeta.candidatesTokenCount : 0;

  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
}

/** Translate Gemini SSE chunks into OpenAI SSE chunks. */
export async function* openAIEventsFromGemini(response: Response, modelId: string): AsyncGenerator<string> {
  const bodyStream = response.body;
  if (bodyStream === null) return;
  const reader = bodyStream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  let promptTokens = 0;
  let completionTokens = 0;

  const sendChunk = (content: string | null, toolCalls: Json[] | null, finishReason: string | null): string => {
    const delta: Json = {};
    if (content !== null) delta.content = content;
    if (toolCalls !== null) delta.tool_calls = toolCalls;
    const chunk = {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    return `data: ${JSON.stringify(chunk)}\n\n`;
  };

  const parseFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usageMetadata)) {
        if (typeof chunk.usageMetadata.promptTokenCount === 'number') promptTokens = chunk.usageMetadata.promptTokenCount;
        if (typeof chunk.usageMetadata.candidatesTokenCount === 'number') completionTokens = chunk.usageMetadata.candidatesTokenCount;
      }
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate === null) continue;
      const candContent = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(candContent.parts) ? candContent.parts : [];

      let text = '';
      const toolCalls: Json[] = [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') text += part.text;
        else if (isRecord(part.functionCall)) {
          const fc = part.functionCall;
          const name = typeof fc.name === 'string' ? fc.name : '';
          const args = isRecord(fc.args) ? fc.args : {};
          toolCalls.push({
            id: `call_${name}_${toolCalls.length}`,
            type: 'function',
            function: { name, arguments: JSON.stringify(args) },
          });
        }
      }

      const finishReasonRaw = stringOf(candidate.finishReason);
      let finishReason: string | null = null;
      if (finishReasonRaw !== '') {
        if (finishReasonRaw === 'MAX_TOKENS') finishReason = 'length';
        else if (finishReasonRaw === 'SAFETY') finishReason = 'content_filter';
        else finishReason = 'stop';
        if (toolCalls.length > 0) finishReason = 'tool_calls';
      }

      if (text !== '' || toolCalls.length > 0 || finishReason !== null) {
        out.push(sendChunk(text !== '' ? text : null, toolCalls.length > 0 ? toolCalls : null, finishReason));
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const ev of parseFrame(frame)) yield ev;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const ev of parseFrame(buffer)) yield ev;
    }
  } finally {
    reader.releaseLock();
  }

  // Final usage chunk & [DONE]
  const usageChunk = {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
  yield `data: ${JSON.stringify(usageChunk)}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** Translate Gemini SSE chunks into Anthropic SSE chunks. */
export async function* anthropicEventsFromGemini(response: Response, modelId: string): AsyncGenerator<string> {
  // We can pipe Gemini SSE through OpenAI chunks generator and then convert to Anthropic events
  const openaiStream = openAIEventsFromGemini(response, modelId);
  // Create a mock response with the generated openai chunks
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      for await (const chunk of openaiStream) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  const mockRes = new Response(stream);
  yield* anthropicEventsFromOpenAI(mockRes, modelId);
}
