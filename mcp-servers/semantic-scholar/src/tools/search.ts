import { client } from "../api/client.js";

export const searchPapersSchema = {
  name: "search_papers",
  description:
    "Search for academic papers on Semantic Scholar. Returns paper metadata including title, authors, abstract, citation count, and external IDs (DOI, ArXiv).",
  inputSchema: {
    type: "object" as const,
    properties: {
      query: {
        type: "string",
        description: "Search query (e.g., 'normalizing flow anomaly detection')",
      },
      limit: {
        type: "number",
        description: "Number of results to return (default: 10, max: 100)",
      },
      year: {
        type: "string",
        description: "Filter by year or year range (e.g., '2023' or '2020-2024')",
      },
      fieldsOfStudy: {
        type: "array",
        items: { type: "string" },
        description:
          "Filter by fields of study (e.g., ['Computer Science', 'Medicine'])",
      },
      openAccessOnly: {
        type: "boolean",
        description: "Only return papers with open access PDF",
      },
    },
    required: ["query"],
  },
};

export async function searchPapers(args: {
  query: string;
  limit?: number;
  year?: string;
  fieldsOfStudy?: string[];
  openAccessOnly?: boolean;
}) {
  const result = await client.searchPapers({
    query: args.query,
    limit: args.limit,
    year: args.year,
    fieldsOfStudy: args.fieldsOfStudy,
    openAccessOnly: args.openAccessOnly,
  });

  return {
    total: result.total,
    count: result.data.length,
    papers: result.data,
  };
}
