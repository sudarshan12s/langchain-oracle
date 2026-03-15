/* eslint-disable no-process-env */
import { test, expect } from "vitest";
import oracledb from "oracledb";
import { OracleSummary } from "../index.js";
import { tool } from "@langchain/core/tools";
import { RunnableSequence } from "@langchain/core/runnables";
import { z } from "zod";

test("Test summary with database", async () => {
  const text =
    "The tower is 324 meters (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 meters (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 meters. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 meters (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.";
  const pref = { provider: "database", gLevel: "S" };

  const connection = await oracledb.getConnection({
    user: process.env.ORACLE_USERNAME,
    password: process.env.ORACLE_PASSWORD,
    connectString: process.env.ORACLE_DSN,
  });

  try {
    const model = new OracleSummary(connection, pref);
    const directSummary = await model.getSummary(text);
    expect(directSummary.length).toBeGreaterThan(1);

    const summarizeTool = tool(
      async ({ text }: { text: string }) => model.getSummary(text),
      {
        name: "oracle_summary",
        description: "Summarize Oracle-sourced documents.",
        schema: z.object({
          text: z.string().describe("Full text to summarize"),
        }),
      },
    );

    const toolSummary = await summarizeTool.invoke({ text });
    expect(toolSummary).toBe(directSummary);

    const summarizeChain = RunnableSequence.from([
      (input: string) => ({ text: input }),
      summarizeTool,
    ]);
    const chainSummary = await summarizeChain.invoke(text);
    expect(chainSummary).toBe(directSummary);
  } finally {
    await connection.close();
  }
});

test("Test summary with third-party", async () => {
  const text =
    "The tower is 324 meters (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 meters (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 meters. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 meters (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.";
  const pref = {
    provider: "ocigenai",
    credential_name: process.env.DEMO_CREDENTIAL,
    url: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
    model: "cohere.command-r-08-2024",
    chatRequest: {
      maxTokens: 256,
    },
  };

  const connection = await oracledb.getConnection({
    user: process.env.ORACLE_USERNAME,
    password: process.env.ORACLE_PASSWORD,
    connectString: process.env.ORACLE_DSN,
  });

  try {
    const model = new OracleSummary(connection, pref);
    const output = await model.getSummary(text);
    expect(output.length).toBeGreaterThan(1);
  } finally {
    await connection.close();
  }
});
