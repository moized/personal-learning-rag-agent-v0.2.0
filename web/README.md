# LearnOS Web

Vite + React + TypeScript client for the Personal Learning Knowledge Agent.

The UI is source-first: add one item now and another later. Files, pasted transcripts/text, and YouTube are separate supported inputs. Multiple files are supported but never required. Metadata is optional.

Install Node.js 20+, then run npm install and npm run dev. The backend defaults to http://localhost:8000; set VITE_API_URL to change it.

The visual system follows the current shadcn/ui Base/Nova conventions while keeping the dependency surface small.