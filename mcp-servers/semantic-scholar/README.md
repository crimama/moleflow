# Semantic Scholar MCP Server

Semantic Scholar API를 활용한 학술 논문 검색 MCP 서버입니다.

## 기능

| Tool | 설명 |
|------|------|
| `search_papers` | 키워드로 논문 검색 |
| `get_paper` | 논문 상세 정보 조회 (DOI, ArXiv ID 지원) |
| `get_citations` | 특정 논문을 인용한 논문들 조회 |
| `get_references` | 특정 논문의 참고문헌 조회 |
| `get_author` | 저자 정보 및 논문 목록 조회 |

## 설치 및 빌드

```bash
cd mcp-servers/semantic-scholar
npm install
npm run build
```

## Claude Code 연동

`~/.claude.json` 파일에 추가:

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "node",
      "args": ["/Volume/MoLeFlow/mcp-servers/semantic-scholar/dist/index.js"]
    }
  }
}
```

## 사용 예시

### 논문 검색
```
"normalizing flow anomaly detection 논문 찾아줘"
→ search_papers 호출
```

### 논문 상세 조회
```
"arXiv:2103.00020 논문 정보 알려줘"
→ get_paper 호출
```

### 인용 논문 찾기
```
"이 논문을 인용한 최신 논문들 보여줘"
→ get_citations 호출
```

### BibTeX 변환
```
"이 논문 BibTeX로 변환해줘"
→ Claude가 JSON 데이터를 BibTeX 형식으로 변환
```

## API Key (선택사항)

Rate limit을 완화하려면 환경변수로 API key를 설정할 수 있습니다:

```bash
export SEMANTIC_SCHOLAR_API_KEY=your_api_key
```

무료로도 사용 가능합니다 (100 requests/5분).

## 반환 데이터 예시

```json
{
  "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
  "title": "Attention Is All You Need",
  "authors": [{"name": "Ashish Vaswani", "authorId": "..."}],
  "year": 2017,
  "citationCount": 90000,
  "abstract": "...",
  "venue": "NeurIPS",
  "openAccessPdf": {"url": "..."},
  "externalIds": {"DOI": "...", "ArXiv": "1706.03762"}
}
```
