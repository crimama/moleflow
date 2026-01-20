#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { searchPapersSchema, searchPapers } from "./tools/search.js";
import { getPaperSchema, getPaper } from "./tools/paper.js";
import {
  getCitationsSchema,
  getCitations,
  getReferencesSchema,
  getReferences,
} from "./tools/citations.js";
import { getAuthorSchema, getAuthor } from "./tools/author.js";

const server = new Server(
  {
    name: "semantic-scholar-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      searchPapersSchema,
      getPaperSchema,
      getCitationsSchema,
      getReferencesSchema,
      getAuthorSchema,
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let result: unknown;

    switch (name) {
      case "search_papers":
        result = await searchPapers(args as Parameters<typeof searchPapers>[0]);
        break;
      case "get_paper":
        result = await getPaper(args as Parameters<typeof getPaper>[0]);
        break;
      case "get_citations":
        result = await getCitations(args as Parameters<typeof getCitations>[0]);
        break;
      case "get_references":
        result = await getReferences(args as Parameters<typeof getReferences>[0]);
        break;
      case "get_author":
        result = await getAuthor(args as Parameters<typeof getAuthor>[0]);
        break;
      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ error: message }, null, 2),
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Semantic Scholar MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
