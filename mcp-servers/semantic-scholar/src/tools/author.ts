import { client } from "../api/client.js";

export const getAuthorSchema = {
  name: "get_author",
  description:
    "Get information about an author including their name, affiliations, paper count, citation count, and h-index.",
  inputSchema: {
    type: "object" as const,
    properties: {
      authorId: {
        type: "string",
        description: "Semantic Scholar author ID",
      },
      includePapers: {
        type: "boolean",
        description: "Whether to include the author's papers (default: false)",
      },
      paperLimit: {
        type: "number",
        description: "Number of papers to return if includePapers is true (default: 100)",
      },
    },
    required: ["authorId"],
  },
};

export async function getAuthor(args: {
  authorId: string;
  includePapers?: boolean;
  paperLimit?: number;
}) {
  const author = await client.getAuthor(args.authorId);

  if (args.includePapers) {
    const papers = await client.getAuthorPapers(args.authorId, args.paperLimit);
    return {
      ...author,
      papers: papers.data,
    };
  }

  return author;
}
