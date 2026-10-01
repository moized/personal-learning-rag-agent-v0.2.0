import type { Metadata } from "next"
import "./globals.css"
export const metadata: Metadata={title:"LearnOS — Personal Learning Knowledge Agent",description:"Organize learning sources, build a curriculum, and study with grounded RAG."}
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}