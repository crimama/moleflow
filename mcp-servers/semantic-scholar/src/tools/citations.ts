import { client } from "../api/client.js";

export const getCitationsSchema = {
  name: "get_citations",
  description:
    "Get papers that cite a specific paper. Useful for finding follow-up work and seeing how a paper has influenced the field.",
  inputSchema: {
    type: "object" as const,
    properties: {
      paperId: {
        type: "string",
        description: "Semantic Scholar paper ID",
      },
      limit: {
        type: "number",
        description: "Number of citing papers to return (default: 50, max: 1000)",
      },
    },
    required: ["paperId"],
  },
};

export async function getCitations(args: { paperId: string; limit?: number }) {
  const result = await client.getCitations(args.paperId, args.limit);

  return {
    count: result.data.length,
    citations: result.data.map((item) => item.citingPaper),
  };
}

export const getReferencesSchema = {
  name: "get_references",
  description:
    "Get papers that a specific paper cites (its bibliography/references). Useful for finding foundational work.",
  inputSchema: {
    type: "object" as const,
    properties: {
      paperId: {
        type: "string",
        description: "Semantic Scholar paper ID",
      },
      limit: {
        type: "number",
        description: "Number of references to return (default: 50, max: 1000)",
      },
    },
    required: ["paperId"],
  },
};

export async function getReferences(args: { paperId: string; limit?: number }) {
  const result = await client.getReferences(args.paperId, args.limit);

  return {
    count: result.data.length,
    references: result.data.map((item) => item.citedPaper),
  };
}
