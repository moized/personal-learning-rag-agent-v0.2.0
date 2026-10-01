# LearnOS Web

Modern Next.js frontend for Personal Learning Knowledge Agent v0.3.

## UX
Overview → Library → Processing → Study.

The Add Material flow supports files, pasted transcripts, Python-dict/JSON content, and YouTube. Advanced metadata stays collapsed. Background processing is visible but not exposed as technical controls.

## Run
npm install
npm run dev

Set NEXT_PUBLIC_API_URL when FastAPI is not at http://localhost:8000.

Stack: Next.js App Router, TypeScript, Tailwind v4, shadcn/ui conventions with Base UI, Lucide.