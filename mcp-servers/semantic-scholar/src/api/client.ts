const BASE_URL = "https://api.semanticscholar.org/graph/v1";

// Fields to request from API
const PAPER_FIELDS = [
  "paperId",
  "title",
  "abstract",
  "year",
  "authors",
  "venue",
  "citationCount",
  "referenceCount",
  "isOpenAccess",
  "openAccessPdf",
  "fieldsOfStudy",
  "publicationDate",
  "externalIds",
  "url",
].join(",");

const AUTHOR_FIELDS = [
  "authorId",
  "name",
  "affiliations",
  "paperCount",
  "citationCount",
  "hIndex",
].join(",");

export interface Paper {
  paperId: string;
  title: string;
  abstract?: string;
  year?: number;
  authors: { authorId: string; name: string }[];
  venue?: string;
  citationCount?: number;
  referenceCount?: number;
  isOpenAccess?: boolean;
  openAccessPdf?: { url: string };
  fieldsOfStudy?: string[];
  publicationDate?: string;
  externalIds?: {
    DOI?: string;
    ArXiv?: string;
    PubMed?: string;
    DBLP?: string;
  };
  url?: string;
}

export interface Author {
  authorId: string;
  name: string;
  affiliations?: string[];
  paperCount?: number;
  citationCount?: number;
  hIndex?: number;
  papers?: Paper[];
}

export interface SearchParams {
  query: string;
  limit?: number;
  year?: string;
  fieldsOfStudy?: string[];
  openAccessOnly?: boolean;
}

class SemanticScholarClient {
  private apiKey?: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.SEMANTIC_SCHOLAR_API_KEY;
  }

  private async fetch<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    const url = new URL(`${BASE_URL}${endpoint}`);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value) url.searchParams.append(key, value);
      });
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["x-api-key"] = this.apiKey;
    }

    const response = await fetch(url.toString(), { headers });

    if (!response.ok) {
      if (response.status === 429) {
        throw new Error("Rate limit exceeded. Please wait a moment and try again.");
      }
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json() as Promise<T>;
  }

  async searchPapers(params: SearchParams): Promise<{ data: Paper[]; total: number }> {
    const { query, limit = 10, year, fieldsOfStudy, openAccessOnly } = params;

    const searchParams: Record<string, string> = {
      query,
      limit: String(Math.min(limit, 100)),
      fields: PAPER_FIELDS,
    };

    if (year) {
      searchParams.year = year;
    }
    if (fieldsOfStudy && fieldsOfStudy.length > 0) {
      searchParams.fieldsOfStudy = fieldsOfStudy.join(",");
    }
    if (openAccessOnly) {
      searchParams.openAccessPdf = "";
    }

    const result = await this.fetch<{ data: Paper[]; total: number }>(
      "/paper/search",
      searchParams
    );

    return result;
  }

  async getPaper(id: string): Promise<Paper> {
    // Handle different ID formats
    let paperId = id;

    // If it looks like a DOI
    if (id.startsWith("10.") || id.includes("doi.org")) {
      paperId = `DOI:${id.replace(/.*doi\.org\//, "")}`;
    }
    // If it looks like an arXiv ID
    else if (id.includes("arxiv.org") || /^\d{4}\.\d{4,5}/.test(id)) {
      paperId = `ARXIV:${id.replace(/.*abs\//, "").replace(/v\d+$/, "")}`;
    }

    const result = await this.fetch<Paper>(`/paper/${encodeURIComponent(paperId)}`, {
      fields: PAPER_FIELDS,
    });

    return result;
  }

  async getCitations(paperId: string, limit = 50): Promise<{ data: { citingPaper: Paper }[] }> {
    const result = await this.fetch<{ data: { citingPaper: Paper }[] }>(
      `/paper/${encodeURIComponent(paperId)}/citations`,
      {
        fields: PAPER_FIELDS,
        limit: String(Math.min(limit, 1000)),
      }
    );

    return result;
  }

  async getReferences(paperId: string, limit = 50): Promise<{ data: { citedPaper: Paper }[] }> {
    const result = await this.fetch<{ data: { citedPaper: Paper }[] }>(
      `/paper/${encodeURIComponent(paperId)}/references`,
      {
        fields: PAPER_FIELDS,
        limit: String(Math.min(limit, 1000)),
      }
    );

    return result;
  }

  async getAuthor(authorId: string): Promise<Author> {
    const result = await this.fetch<Author>(`/author/${encodeURIComponent(authorId)}`, {
      fields: AUTHOR_FIELDS,
    });

    return result;
  }

  async getAuthorPapers(authorId: string, limit = 100): Promise<{ data: Paper[] }> {
    const result = await this.fetch<{ data: Paper[] }>(
      `/author/${encodeURIComponent(authorId)}/papers`,
      {
        fields: PAPER_FIELDS,
        limit: String(Math.min(limit, 1000)),
      }
    );

    return result;
  }
}

export const client = new SemanticScholarClient();
export { SemanticScholarClient };
