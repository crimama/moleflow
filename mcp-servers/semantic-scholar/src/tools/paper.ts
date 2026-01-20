import { client } from "../api/client.js";

export const getPaperSchema = {
  name: "get_paper",
  description:
    "Get detailed information about a specific paper. Accepts Semantic Scholar paper ID, DOI, ArXiv ID, or URL.",
  inputSchema: {
    type: "object" as const,
    properties: {
      id: {
        type: "string",
        description:
          "Paper identifier: Semantic Scholar ID, DOI (e.g., '10.1234/example'), ArXiv ID (e.g., '2103.00020'), or URL",
      },
    },
    required: ["id"],
  },
};

export async function getPaper(args: { id: string }) {
  const paper = await client.getPaper(args.id);
  return paper;
}
