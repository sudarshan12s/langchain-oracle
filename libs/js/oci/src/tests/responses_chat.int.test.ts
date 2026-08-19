/* eslint-disable no-process-env */

import { expect, test } from "vitest";

import { OciGenAiResponsesChat } from "../responses_chat.js";
import { OciGenAiNewClientAuthType } from "../types.js";

// The Responses API requires a Generative AI project in addition to normal
// OCI credentials, so this billable test stays opt-in.
const projectId = process.env.OCI_GENAI_RESPONSES_INTEGRATION_TESTS_PROJECT_ID;
const model =
  process.env.OCI_GENAI_RESPONSES_INTEGRATION_TESTS_MODEL ??
  "xai.grok-4.20-multi-agent-0309";
const regionId = process.env.OCI_REGION ?? "us-chicago-1";
const configFilePath = process.env.OCI_CONFIG_FILE;
const configProfile = process.env.OCI_CONFIG_PROFILE;

test.skipIf(!projectId)(
  "OCI Responses API invoke",
  async () => {
    const chat = new OciGenAiResponsesChat({
      model,
      projectId: projectId!,
      endpoint: process.env.OCI_OPENAI_ENDPOINT,
      newClientParams: {
        authType: OciGenAiNewClientAuthType.ConfigFile,
        regionId,
        authParams:
          configFilePath || configProfile
            ? {
                clientConfigFilePath: configFilePath ?? "",
                clientProfile: configProfile ?? "DEFAULT",
              }
            : undefined,
      },
    });

    const response = await chat.invoke("Reply with exactly: hello");
    expect(typeof response.content).toBe("string");
    expect(response.content.length).toBeGreaterThan(0);
  },
  100_000
);
