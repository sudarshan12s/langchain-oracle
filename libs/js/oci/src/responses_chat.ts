import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  ToolMessage as LangChainToolMessage,
  type ToolCall,
  type UsageMetadata,
} from "@langchain/core/messages";
import {
  BaseChatModel,
  type BindToolsInput,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
  type LangSmithParams,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import {
  ChatGenerationChunk,
  type ChatGeneration,
  type ChatResult,
} from "@langchain/core/outputs";
import { RunnableBinding, type Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import type { RequestSigner } from "oci-common";

import { OciGenAiSdkClient } from "./oci_genai_sdk_client.js";
import { JsonServerEventsIterator } from "./server_events_iterator.js";
import type { OciGenAiNewClientParams } from "./types.js";

const DEFAULT_REGION_ID = "us-chicago-1";

/** OCI OpenAI-compatible Responses API invocation options. */
export interface OciGenAiResponsesCallOptions extends BaseChatModelCallOptions {
  temperature?: number;
  maxOutputTokens?: number;
  tools?: unknown[];
  toolChoice?: unknown;
  conversationId?: string;
  store?: boolean;
}

/**
 * Parameters for OCI's OpenAI-compatible Responses API.
 *
 * `projectId` is required by current OCI OpenAI-compatible endpoints.
 * `conversationStoreId` is retained only for legacy OCI deployments; prefer
 * the `conversationId` call option with the OCI Conversations API.
 */
export interface OciGenAiResponsesChatParams extends BaseChatModelParams {
  model: string;
  projectId: string;
  endpoint?: string;
  newClientParams?: OciGenAiNewClientParams;
  conversationStoreId?: string;
  requestSigner?: RequestSigner;
  fetch?: typeof globalThis.fetch;
}

interface OciResponsesUsage {
  input_tokens?: unknown;
  output_tokens?: unknown;
  total_tokens?: unknown;
}

interface OciResponsesToolCall {
  type?: unknown;
  name?: unknown;
  arguments?: unknown;
  call_id?: unknown;
}

interface OciResponsesResponse {
  id?: unknown;
  model?: unknown;
  status?: unknown;
  output?: unknown;
  usage?: OciResponsesUsage;
}

/**
 * Chat model for OCI's OpenAI-compatible Responses API.
 *
 * This is intentionally separate from OciGenAiGenericChat: Responses uses a
 * different endpoint, message schema, state model, and stream-event protocol
 * than OCI's native `/20231130/actions/chat` SDK endpoint.
 */
export class OciGenAiResponsesChat extends BaseChatModel<OciGenAiResponsesCallOptions> {
  private readonly _params: OciGenAiResponsesChatParams;

  private _requestSignerPromise: Promise<RequestSigner> | undefined;

  constructor(params: OciGenAiResponsesChatParams) {
    super(params);
    if (!params.model) {
      throw new Error("Responses API model is required");
    }
    if (!params.projectId) {
      throw new Error("OCI Responses API projectId is required");
    }
    this._params = params;
  }

  _llmType(): string {
    return "oci_genai_responses";
  }

  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const response = await this._request(messages, options, false);
    if (OciGenAiResponsesChat._isReadableStream(response)) {
      throw new Error(
        "OCI Responses API returned a stream for a non-streaming call"
      );
    }
    const content = OciGenAiResponsesChat._getResponseText(response);
    const toolCalls = OciGenAiResponsesChat._getToolCalls(response);
    const usageMetadata = OciGenAiResponsesChat._toUsageMetadata(
      response.usage
    );
    const responseMetadata = OciGenAiResponsesChat._responseMetadata(response);
    const message = new AIMessage({
      content,
      tool_calls: toolCalls,
      usage_metadata: usageMetadata,
      response_metadata: responseMetadata,
    });
    const generation: ChatGeneration = {
      message,
      text: content,
      generationInfo: responseMetadata,
    };
    return {
      generations: [generation],
      llmOutput: { ...responseMetadata, usage: usageMetadata },
    };
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const stream = await this._request(messages, options, true);
    if (!OciGenAiResponsesChat._isReadableStream(stream)) {
      throw new Error("OCI Responses API did not return a response stream");
    }

    for await (const event of new JsonServerEventsIterator(stream)) {
      const chunk = OciGenAiResponsesChat._parseStreamEvent(event);
      if (!chunk) {
        continue;
      }
      const generation = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: chunk.text ?? "",
          usage_metadata: chunk.usageMetadata,
          response_metadata: chunk.responseMetadata,
        }),
        text: chunk.text ?? "",
      });
      if (chunk.text) {
        await runManager?.handleLLMNewToken(chunk.text);
      }
      yield generation;
    }
  }

  override getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    return {
      ls_provider: "oci_genai_responses",
      ls_model_name: this._params.model,
      ls_model_type: "chat",
      ls_temperature: options.temperature ?? 0,
      ls_max_tokens: options.maxOutputTokens ?? 0,
      ls_stop: options.stop ?? [],
    };
  }

  /** Binds OpenAI-format function definitions accepted by the Responses API. */
  bindTools(
    tools: BindToolsInput[],
    kwargs: OciGenAiResponsesCallOptions = {}
  ): Runnable<
    BaseLanguageModelInput,
    AIMessageChunk,
    OciGenAiResponsesCallOptions
  > {
    return new RunnableBinding({
      bound: this,
      kwargs: {
        ...kwargs,
        tools: tools.map(convertToOpenAITool),
      },
      config: {},
    });
  }

  private async _request(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    stream: boolean
  ): Promise<OciResponsesResponse | ReadableStream<Uint8Array>> {
    const body = JSON.stringify({
      model: this._params.model,
      input: OciGenAiResponsesChat._convertMessages(messages),
      stream,
      ...(options.temperature !== undefined
        ? { temperature: options.temperature }
        : {}),
      ...(options.maxOutputTokens !== undefined
        ? { max_output_tokens: options.maxOutputTokens }
        : {}),
      ...(options.tools !== undefined ? { tools: options.tools } : {}),
      ...(options.toolChoice !== undefined
        ? { tool_choice: options.toolChoice }
        : {}),
      ...(options.conversationId !== undefined
        ? { conversation: options.conversationId }
        : {}),
      ...(options.store !== undefined ? { store: options.store } : {}),
    });

    return this.caller.call(async () => {
      const response = await this._fetch(body);
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(
          `OCI Responses API request failed (${response.status}): ${detail}`
        );
      }
      if (stream) {
        if (!response.body) {
          throw new Error("OCI Responses API response has no stream body");
        }
        return response.body;
      }
      return (await response.json()) as OciResponsesResponse;
    });
  }

  private async _fetch(body: string): Promise<Response> {
    const url = `${this._baseUrl()}/responses`;
    const headers = new Headers({
      Accept: "application/json",
      "Content-Type": "application/json",
      "OpenAI-Project": this._params.projectId,
    });
    if (this._params.conversationStoreId) {
      headers.set(
        "opc-conversation-store-id",
        this._params.conversationStoreId
      );
    }
    await (
      await this._getRequestSigner()
    ).signHttpRequest({
      method: "POST",
      uri: url,
      headers,
      body,
    });
    return (this._params.fetch ?? globalThis.fetch)(url, {
      method: "POST",
      headers,
      body,
    });
  }

  private _baseUrl(): string {
    const endpoint =
      this._params.endpoint ??
      this._params.newClientParams?.serviceEndpoint ??
      `https://inference.generativeai.${
        this._params.newClientParams?.regionId ?? DEFAULT_REGION_ID
      }.oci.oraclecloud.com`;
    const normalized = endpoint.replace(/\/$/, "");
    return normalized.endsWith("/openai/v1")
      ? normalized
      : `${normalized}/openai/v1`;
  }

  private async _getRequestSigner(): Promise<RequestSigner> {
    if (this._params.requestSigner) {
      return this._params.requestSigner;
    }
    if (!this._requestSignerPromise) {
      this._requestSignerPromise = OciGenAiSdkClient.createRequestSigner({
        newClientParams: this._params.newClientParams,
      });
    }
    return this._requestSignerPromise;
  }

  private static _convertMessages(messages: BaseMessage[]): unknown[] {
    if (messages.length === 0) {
      throw new Error("No messages provided");
    }
    return messages.flatMap<unknown>((message) => {
      const content = OciGenAiResponsesChat._contentToText(message.content);
      switch (message.getType()) {
        case "system":
          return { role: "developer", content };
        case "human":
          return { role: "user", content };
        case "ai": {
          const toolCalls =
            (
              message as {
                tool_calls?: Array<{
                  id?: string;
                  name: string;
                  args: unknown;
                }>;
              }
            ).tool_calls ?? [];
          for (const toolCall of toolCalls) {
            if (!toolCall.id) {
              throw new Error(
                `LangChain tool call '${toolCall.name}' did not contain a tool call id`
              );
            }
          }
          if (toolCalls.length === 0) {
            return { role: "assistant", content };
          }
          return [
            ...(content ? [{ role: "assistant", content }] : []),
            ...toolCalls.map((toolCall) => ({
              type: "function_call",
              call_id: toolCall.id,
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.args ?? {}),
            })),
          ];
        }
        case "tool": {
          const toolCallId = (message as LangChainToolMessage).tool_call_id;
          if (!toolCallId) {
            throw new Error("ToolMessage did not contain a tool call id");
          }
          return {
            type: "function_call_output",
            call_id: toolCallId,
            output: content,
          };
        }
        default:
          throw new Error(
            `Message type '${message.getType()}' is not supported`
          );
      }
    });
  }

  private static _contentToText(content: BaseMessage["content"]): string {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      const textBlocks = content.filter(
        (block): block is { type: "text"; text: string } =>
          block !== null &&
          typeof block === "object" &&
          "type" in block &&
          block.type === "text" &&
          "text" in block &&
          typeof block.text === "string"
      );
      if (textBlocks.length === content.length) {
        return textBlocks.map((block) => block.text).join("");
      }
    }
    throw new Error("Unsupported message content for OCI Responses API");
  }

  private static _getResponseText(response: OciResponsesResponse): string {
    if (!Array.isArray(response.output)) {
      return "";
    }
    return response.output
      .flatMap((item) => {
        if (item === null || typeof item !== "object") {
          return [];
        }
        const { content } = item as { content?: unknown };
        if (!Array.isArray(content)) {
          return [];
        }
        return content.flatMap((part) =>
          part !== null &&
          typeof part === "object" &&
          (part as { type?: unknown }).type === "output_text" &&
          typeof (part as { text?: unknown }).text === "string"
            ? [(part as { text: string }).text]
            : []
        );
      })
      .join("");
  }

  private static _getToolCalls(response: OciResponsesResponse): ToolCall[] {
    if (!Array.isArray(response.output)) {
      return [];
    }
    return response.output.flatMap((item) => {
      const call = item as OciResponsesToolCall;
      if (
        item === null ||
        typeof item !== "object" ||
        call.type !== "function_call"
      ) {
        return [];
      }
      if (
        typeof call.name !== "string" ||
        !call.name ||
        typeof call.call_id !== "string" ||
        !call.call_id
      ) {
        throw new Error(
          "OCI Responses API function call is missing name or call_id"
        );
      }
      let args: unknown = {};
      if (typeof call.arguments === "string") {
        try {
          args = JSON.parse(call.arguments);
        } catch {
          // Retain the valid call and use safe empty arguments for malformed JSON.
          args = {};
        }
      }
      return [
        {
          type: "tool_call" as const,
          id: call.call_id,
          name: call.name,
          args: args !== null && typeof args === "object" ? args : {},
        },
      ];
    });
  }

  private static _toUsageMetadata(
    usage: OciResponsesUsage | undefined
  ): UsageMetadata | undefined {
    if (!usage) {
      return undefined;
    }
    const input = OciGenAiResponsesChat._finiteNumber(usage.input_tokens) ?? 0;
    const output =
      OciGenAiResponsesChat._finiteNumber(usage.output_tokens) ?? 0;
    return {
      input_tokens: input,
      output_tokens: output,
      total_tokens:
        OciGenAiResponsesChat._finiteNumber(usage.total_tokens) ??
        input + output,
    };
  }

  private static _finiteNumber(value: unknown): number | undefined {
    return typeof value === "number" && Number.isFinite(value)
      ? value
      : undefined;
  }

  private static _isReadableStream(
    value: unknown
  ): value is ReadableStream<Uint8Array> {
    return (
      value !== null &&
      typeof value === "object" &&
      "getReader" in value &&
      typeof (value as { getReader?: unknown }).getReader === "function"
    );
  }

  private static _responseMetadata(
    response: OciResponsesResponse
  ): Record<string, unknown> {
    return {
      ...(typeof response.id === "string" ? { response_id: response.id } : {}),
      ...(typeof response.model === "string" ? { model: response.model } : {}),
      ...(typeof response.status === "string"
        ? { status: response.status }
        : {}),
    };
  }

  private static _parseStreamEvent(event: unknown):
    | {
        text?: string;
        usageMetadata?: UsageMetadata;
        responseMetadata?: Record<string, unknown>;
      }
    | undefined {
    if (event === null || typeof event !== "object") {
      return undefined;
    }
    const { type } = event as { type?: unknown };
    if (type === "response.output_text.delta") {
      const { delta } = event as { delta?: unknown };
      return typeof delta === "string" ? { text: delta } : undefined;
    }
    if (type === "response.completed") {
      const { response } = event as { response?: unknown };
      if (response !== null && typeof response === "object") {
        return {
          usageMetadata: OciGenAiResponsesChat._toUsageMetadata(
            (response as OciResponsesResponse).usage
          ),
          responseMetadata: OciGenAiResponsesChat._responseMetadata(
            response as OciResponsesResponse
          ),
        };
      }
    }
    return undefined;
  }
}
