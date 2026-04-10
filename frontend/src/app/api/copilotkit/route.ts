import {
  CopilotRuntime,
  GoogleGenerativeAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const runtime = new CopilotRuntime();

const handler = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  serviceAdapter: new GoogleGenerativeAIAdapter({
    model: "gemini-2.0-flash-exp",
  }),
  endpoint: "/api/copilotkit",
});

export const POST = handler;