import {useEffect,useMemo,useRef,useState,type FormEvent,type KeyboardEvent,type ReactNode} from "react"
import {Brain,Check,ChevronDown,ChevronRight,Copy,Download,FileText,FolderOpen,GraduationCap,Library,LoaderCircle,MessageSquare,Paperclip,Plus,RefreshCw,Send,Sparkles,Trash2,UploadCloud,X} from "lucide-react"
import {api,type ChatTurn,type ConversationSummary,type Course,type Job,type MaintenanceJob,type PasteFormat,type MasterPrompt,type PlanStatus,type PreparedTopic,type Profile,type SessionSize,type Source,type StudyPackExport} from "./lib/api"
import LearnView from "./Learn"

type View="chat"|"library"|"plan"|"learn"
type ChatMessage={id:string;role:"user"|"assistant";content:string;evidenceCount?:number;supplementalSourceCount?:number;supplementalSources?:string[];retryJobId?:number}
type AddMode="file"|"youtube"|"paste"
type OutlineTopic={id:string;parentId:string;pathId:string;kind:"outline"|"study";topic:string;purpose:string;prerequisites:string;groups:string[];why:string;enables:string;sourceIds:number[];sourceRoles:string[];confidence:number}
type PlanRelationships={sources:any[];cross_paths:any[];shared_foundations:string[]}
function normalizePlanRelationships(value:unknown):PlanRelationships{
  const empty:PlanRelationships={sources:[],cross_paths:[],shared_foundations:[]}
  if(Array.isArray(value))return {...empty,sources:value}
  if(!value||typeof value!=="object")return empty
  const data=value as Record<string,unknown>
  return {sources:Array.isArray(data.sources)?data.sources:[],cross_paths:Array.isArray(data.cross_paths)?data.cross_paths:[],shared_foundations:Array.isArray(data.shared_foundations)?data.shared_foundations.filter((item):item is string=>typeof item==="string"):[]}
}

function Button({children,onClick,variant="primary",disabled=false,type="button"}:{children:ReactNode;onClick?:()=>void;variant?:"primary"|"outline"|"ghost"|"danger";disabled?:boolean;type?:"button"|"submit"}){
  return <button className={`btn ${variant}`} type={type} onClick={onClick} disabled={disabled}>{children}</button>
}
function Badge({children}:{children:ReactNode}){return <span className="badge">{children}</span>}
function stageLabel(stage:string){return ({structure:"Preparing source",transcript:"Getting transcript",index:"Indexing content",organize:"Organizing notes",concepts:"Finding key ideas",graph:"Connecting ideas",done:"Ready"} as Record<string,string>)[stage]||"Working"}
function statusLabel(status:string){return status==="done"?"Ready":status==="needs_input"?"Needs your input":status==="failed"?"Needs attention":status==="retrying"?"Retrying":status==="running"||status==="processing"?"In progress":"Queued"}
function sourceTitle(source:Pick<Source,"name"|"source_type">|Pick<Job,"source_name"|"source_type"|"source_uri">){
  const name="name" in source?source.name:source.source_name
  if(source.source_type!=="youtube")return name||"Source"
  if(name&&!/^https?:\/\//i.test(name))return name
  const uri="source_uri" in source?source.source_uri:name
  try{const url=new URL(uri||"");const playlist=url.searchParams.get("list");const id=url.hostname==="youtu.be"?url.pathname.slice(1):url.searchParams.get("v");return playlist?`YouTube playlist · ${playlist}`:id?`YouTube video · ${id}`:"YouTube source"}catch{return "YouTube source"}
}
function sourceKey(job:Job){
  if(job.source_type!=="youtube")return `source:${job.source_id}`
  try{const url=new URL(job.source_uri||job.source_name||"");const playlist=url.searchParams.get("list");const id=url.hostname==="youtu.be"?url.pathname.slice(1):url.searchParams.get("v");return playlist?`playlist:${playlist}`:id?`video:${id}`:`youtube:${url.hostname}${url.pathname}`}catch{return `source:${job.source_id}`}
}
function latestJobs(jobs:Job[]){const keys=new Set<string>();return jobs.filter(job=>{const key=sourceKey(job);if(keys.has(key))return false;keys.add(key);return true})}
function newId(){return globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random()}`}
function normalizedTopic(topic:string){return topic.trim().toLowerCase().replace(/\s+/g," ")}
function parseOutline(markdown:string):OutlineTopic[]{
  const rows=markdown.split("\n").filter(line=>/^\s*\|/.test(line))
  if(!rows.length)return []
  const cells=(row:string)=>row.trim().replace(/^\||\|$/g,"").split("|").map(value=>value.trim())
  const header=cells(rows[0]).map(value=>value.toLowerCase())
  const column=(...names:string[])=>{const index=header.findIndex(value=>names.includes(value));return index}
  const topicCol=column("topic","title")
  return rows.slice(1).filter(line=>!/^\s*\|?\s*:?-{3,}/.test(line.trim())).flatMap(line=>{
    const values=cells(line)
    const value=(index:number,fallback:number)=>index>=0?values[index]||"":fallback>=0?values[fallback]||"":""
    const topic=value(topicCol,1)
    if(!topic||topic.toLowerCase()==="topic")return []
    const groups=value(column("source groups","groups"),5).split(",").map(item=>item.trim()).filter(item=>item&&item.toLowerCase()!=="none"&&item!=="—")
    const kind=value(column("type","kind"),-1).toLowerCase()==="outline"?"outline":"study"
    return [{id:value(column("id"),0)||topic,parentId:value(column("parent id","parent"),-1),pathId:"main",kind,topic:topic.replace(/\\\|/g,"|"),purpose:value(column("purpose"),2),prerequisites:value(column("prerequisites"),3),groups:groups.length?groups:["Other topics"],why:"",enables:"",sourceIds:[],sourceRoles:[],confidence:0}]
  })
}

function inlineMarkdown(text:string,keyPrefix:string):ReactNode[]{
  const nodes:ReactNode[]=[]
  const pattern=/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\)|https?:\/\/[^\s<]+)/g
  let last=0,match:RegExpExecArray|null,index=0
  while((match=pattern.exec(text))){
    if(match.index>last)nodes.push(text.slice(last,match.index))
    const token=match[0],key=`${keyPrefix}-${index++}`
    if(token.startsWith("`"))nodes.push(<code key={key}>{token.slice(1,-1)}</code>)
    else if(token.startsWith("**"))nodes.push(<strong key={key}>{token.slice(2,-2)}</strong>)
    else if(token.startsWith("*"))nodes.push(<em key={key}>{token.slice(1,-1)}</em>)
    else if(token.startsWith("[")){const parsed=token.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);nodes.push(parsed?<a key={key} href={parsed[2]} target="_blank" rel="noreferrer">{parsed[1]}</a>:token)}
    else {const trailing=token.match(/[.,!?;:]+$/)?.[0]||"";const href=trailing?token.slice(0,-trailing.length):token;nodes.push(<span key={key}><a href={href} target="_blank" rel="noreferrer">{href}</a>{trailing}</span>)}
    last=pattern.lastIndex
  }
  if(last<text.length)nodes.push(text.slice(last))
  return nodes
}

function MarkdownText({text}:{text:string}){
  const lines=text.replace(/\r\n?/g,"\n").split("\n")
  const blocks:ReactNode[]=[]
  let cursor=0,blockKey=0
  const isBlockStart=(line:string)=>/^\s*(?:#{1,4}\s|```|>\s?|[-*+]\s+|\d+[.)]\s+|\|.*\|\s*$|---+\s*$)/.test(line)
  while(cursor<lines.length){
    const line=lines[cursor]
    if(!line.trim()){cursor++;continue}
    const key=`md-${blockKey++}`
    if(line.trim().startsWith("```")){
      const language=line.trim().slice(3).trim();cursor++
      const code:string[]=[]
      while(cursor<lines.length&&!lines[cursor].trim().startsWith("```"))code.push(lines[cursor++])
      if(cursor<lines.length)cursor++
      blocks.push(<pre key={key}><code className={language?`language-${language}`:undefined}>{code.join("\n")}</code></pre>);continue
    }
    const heading=line.match(/^\s*(#{1,4})\s+(.+)$/)
    if(heading){const content=inlineMarkdown(heading[2],key);const level=heading[1].length;blocks.push(level===1||level===2?<h2 key={key}>{content}</h2>:<h3 key={key}>{content}</h3>);cursor++;continue}
    if(/^\s*---+\s*$/.test(line)){blocks.push(<hr key={key}/>);cursor++;continue}
    if(/^\s*\|.*\|\s*$/.test(line)){
      const tableLines:string[]=[]
      while(cursor<lines.length&&/^\s*\|.*\|\s*$/.test(lines[cursor]))tableLines.push(lines[cursor++])
      const cells=(row:string)=>row.trim().replace(/^\||\|$/g,"").split("|").map(cell=>cell.trim())
      const header=cells(tableLines[0]||"")
      const hasDivider=tableLines[1]&&/^\s*\|?\s*:?-{3,}/.test(tableLines[1])
      const bodyRows=tableLines.slice(hasDivider?2:1).map(cells)
      blocks.push(<div className="markdown-table-wrap" key={key}><table><thead><tr>{header.map((cell,i)=><th key={i}>{inlineMarkdown(cell,`${key}-h${i}`)}</th>)}</tr></thead><tbody>{bodyRows.map((row,r)=><tr key={r}>{header.map((_,i)=><td key={i}>{inlineMarkdown(row[i]||"",`${key}-r${r}c${i}`)}</td>)}</tr>)}</tbody></table></div>);continue
    }
    if(/^\s*>/.test(line)){
      const quote:string[]=[]
      while(cursor<lines.length&&/^\s*>/.test(lines[cursor]))quote.push(lines[cursor++].replace(/^\s*>\s?/,""))
      blocks.push(<blockquote key={key}>{quote.map((part,i)=><p key={i}>{inlineMarkdown(part,`${key}-${i}`)}</p>)}</blockquote>);continue
    }
    const listMatch=line.match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)/)
    if(listMatch){
      const ordered=/^\s*\d/.test(line),items:string[]=[]
      const listPattern=ordered?/^\s*\d+[.)]\s+/:/^\s*[-*+]\s+/
      while(cursor<lines.length){const item=lines[cursor].match(listPattern);if(!item)break;items.push(lines[cursor++].replace(listPattern,""))}
      const List=ordered?"ol":"ul"
      blocks.push(<List key={key}>{items.map((item,i)=><li key={i}>{inlineMarkdown(item,`${key}-${i}`)}</li>)}</List>);continue
    }
    const paragraph=[line.trim()];cursor++
    while(cursor<lines.length&&lines[cursor].trim()&&!isBlockStart(lines[cursor]))paragraph.push(lines[cursor++].trim())
    blocks.push(<p key={key}>{paragraph.map((part,i)=><span key={i}>{i>0&&" "}{inlineMarkdown(part,`${key}-${i}`)}</span>)}</p>)
  }
  return <div className="markdown-body">{blocks}</div>
}

function plural(count:number,word:string){return `${count.toLocaleString()} ${word}${count===1?"":"s"}`}
const TUTOR_SETUP:Record<string,string>={
  general:"Paste it into your AI tool's project or custom instructions, then open a new chat there for each study session.",
  chatgpt:"In ChatGPT, create a Project and paste it into the project's instructions, then open a new chat in the project for each study session.",
  claude:"In Claude, create a Project and paste it into the project's instructions, then open a new chat in the project for each study session.",
  gemini:"In Gemini, create a Gem with it as the Gem's instructions, then start a new chat with the Gem for each study session.",
}
const SIZE_TOKENS:Record<SessionSize,number>={compact:4000,standard:8000,large:16000,xl:32000}
function remembered(key:string,allowed:string[],fallback:string){try{const value=localStorage.getItem(key);return value&&allowed.includes(value)?value:fallback}catch{return fallback}}
function remember(key:string,value:string){try{localStorage.setItem(key,value)}catch{/* not remembered in this browser */}}
function fileSlug(title:string){return title.normalize("NFKD").replace(/\p{M}+/gu,"").replace(/[^\p{L}\p{N}]+/gu,"-").replace(/^-+|-+$/g,"").slice(0,80).toLowerCase()||"study-unit"}
function downloadText(name:string,text:string){const url=URL.createObjectURL(new Blob([text],{type:"text/markdown;charset=utf-8"}));const link=document.createElement("a");link.href=url;link.download=name;link.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000)}

function CopyAnswer({text,label="Copy",primary=false}:{text:string;label?:string;primary?:boolean}){
  const[copied,setCopied]=useState(false)
  const timer=useRef<number|undefined>(undefined)
  useEffect(()=>()=>window.clearTimeout(timer.current),[])
  const copy=async()=>{
    try{
      if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(text)
      else{const area=document.createElement("textarea");area.value=text;area.style.position="fixed";area.style.opacity="0";document.body.appendChild(area);area.select();const ok=document.execCommand("copy");area.remove();if(!ok)throw new Error("Clipboard unavailable")}
      setCopied(true);timer.current=window.setTimeout(()=>setCopied(false),1800)
    }catch{setCopied(false)}
  }
  return <button type="button" className={primary?"btn primary copy-primary":"copy-answer"} onClick={()=>void copy()} aria-label={copied?"Copied":label} title={copied?"Copied":label}>{copied?<Check size={primary?16:14}/>:<Copy size={primary?16:14}/>}<span>{copied?"Copied":label}</span></button>
}

const VIEWS:View[]=["plan","library","learn","chat"]
function readLocation():{view:View;courseId?:number}{
  const match=window.location.hash.match(/^#\/(\w+)(?:\?space=(\d+))?$/)
  const view=match&&VIEWS.includes(match[1] as View)?match[1] as View:"plan"
  return {view,courseId:match?.[2]?Number(match[2]):undefined}
}
function locationFor(view:View,courseId?:number){return `#/${view}${courseId?`?space=${courseId}`:""}`}

export default function App(){
  const initial=useMemo(readLocation,[])
  const[view,setView]=useState<View>(initial.view)
  const[courses,setCourses]=useState<Course[]>([])
  const[courseId,setCourseId]=useState<number|undefined>(initial.courseId)
  useEffect(()=>{const target=locationFor(view,courseId);if(window.location.hash===target)return;const shown=readLocation()
    // A new place gets a history entry; the same place written differently (or no address yet) is just corrected.
    if(window.location.hash&&(shown.view!==view||shown.courseId!==courseId))window.history.pushState(null,"",target);else window.history.replaceState(null,"",target)},[view,courseId])
  useEffect(()=>{const follow=()=>{const next=readLocation();setView(next.view);setCourseId(next.courseId)};window.addEventListener("popstate",follow);window.addEventListener("hashchange",follow);return()=>{window.removeEventListener("popstate",follow);window.removeEventListener("hashchange",follow)}},[])

  const[sources,setSources]=useState<Source[]>([])
  const[sourcesFor,setSourcesFor]=useState<string|null>(null)  // the space the source list was loaded for
  const[jobs,setJobs]=useState<Job[]>([])
  const[messages,setMessages]=useState<ChatMessage[]>([])
  const[conversations,setConversations]=useState<ConversationSummary[]>([])
  const[conversationId,setConversationId]=useState<string>()
  const conversationIdRef=useRef<string|undefined>(undefined)
  const[busy,setBusy]=useState(false)
  const[retryingJob,setRetryingJob]=useState<number|null>(null)
  const[addOpen,setAddOpen]=useState(false)
  const[deletingSpace,setDeletingSpace]=useState<Course|null>(null)
  const[error,setError]=useState("")
  const[health,setHealth]=useState<any>()
  const[planRefreshKey,setPlanRefreshKey]=useState(0)
  const[planRegenerateKey,setPlanRegenerateKey]=useState(0)
  const previousJobStates=useRef(new Map<number,string>())
  const refreshSequence=useRef(0)
  const chatOpenSequence=useRef(0)
  const[placementPending,setPlacementPending]=useState(false)

  const refresh=async()=>{
    const requestId=++refreshSequence.current
    try{
      setError("")
      const[c,j,h,s,checking]=await Promise.all([api.courses(),api.jobs(),api.health(),api.sources(courseId),api.placementChecksActive().catch(()=>false)])
      if(requestId!==refreshSequence.current)return
      setPlacementPending(checking)
      const terminal=new Set(["done","failed"])
      const becameReady=j.some(job=>previousJobStates.current.get(job.id)!==undefined&&
        !terminal.has(previousJobStates.current.get(job.id)!)&&terminal.has(job.status))
      previousJobStates.current=new Map(j.map(job=>[job.id,job.status]))
      setCourses(c);setJobs(j);setHealth(h);setSources(s);setSourcesFor(courseId===undefined?"all":String(courseId))
      // A space in an old link that no longer exists: show all sources instead (checked against the list just loaded).
      if(courseId!==undefined&&!c.some(course=>course.id===courseId)){window.history.replaceState(null,"",locationFor(readLocation().view));setCourseId(undefined)}
      if(becameReady)setPlanRefreshKey(key=>key+1)
    }catch(e){setError(e instanceof Error?e.message:"Could not connect to LearnOS")}
  }
  useEffect(()=>{refreshSequence.current++;void refresh()},[courseId])
  useEffect(()=>{let current=true;api.conversations().then(async loaded=>{if(!current)return;let rows=loaded;if(!rows.length&&!localStorage.getItem("learnos.chat-migration-v1")){try{const old=JSON.parse(localStorage.getItem("learnos.current-chat")||"[]") as ChatMessage[];if(old.length){const created=await api.createConversation();for(const message of old)await api.addConversationMessage(created.id,{role:message.role,content:message.content},{evidenceCount:message.evidenceCount,supplementalSourceCount:message.supplementalSourceCount,supplementalSources:message.supplementalSources,retryJobId:message.retryJobId});localStorage.setItem("learnos.chat-migration-v1","done");rows=await api.conversations()}}catch{setError("Your earlier chat is still stored in this browser; LearnOS could not migrate it yet.")}}if(!current)return;setConversations(rows);const saved=localStorage.getItem("learnos.active-conversation");const selected=rows.find(row=>row.id===saved)||rows[0];if(selected){conversationIdRef.current=selected.id;setConversationId(selected.id);const full=await api.getConversation(selected.id);if(current)setMessages(full.messages.map(m=>({id:String(m.id),role:m.role,content:m.content,evidenceCount:m.metadata?.evidenceCount,supplementalSourceCount:m.metadata?.supplementalSourceCount,supplementalSources:m.metadata?.supplementalSources,retryJobId:m.metadata?.retryJobId})))}}).catch(e=>setError(e instanceof Error?e.message:"Could not load conversations"));return()=>{current=false}},[])
  useEffect(()=>{if(conversationId)localStorage.setItem("learnos.active-conversation",conversationId)},[conversationId])
  const currentJobs=useMemo(()=>latestJobs(jobs),[jobs])
  const activeJobs=currentJobs.filter(job=>["queued","running","retrying"].includes(job.status))
  const failedJobs=currentJobs.filter(job=>job.status==="failed")
  // Poll while sources are processing or their placement is being checked; read once more when the checks finish,
  // so suggestions made by the last check are shown.
  const polling=activeJobs.length>0||placementPending
  const wasPolling=useRef(false)
  useEffect(()=>{
    if(wasPolling.current&&!polling)void refresh()
    wasPolling.current=polling
    if(!polling)return
    const timer=window.setInterval(()=>void refresh(),3000)
    return()=>window.clearInterval(timer)
  },[polling,courseId])
  useEffect(()=>{if(view==="library")void refresh()},[view])

  const appendAssistant=(content:string,extra:Partial<ChatMessage>={})=>{const message:ChatMessage={id:newId(),role:"assistant",content,...extra};setMessages(previous=>[...previous,message]);const activeId=conversationIdRef.current;if(activeId)void api.addConversationMessage(activeId,{role:"assistant",content},{evidenceCount:extra.evidenceCount,supplementalSourceCount:extra.supplementalSourceCount,supplementalSources:extra.supplementalSources,retryJobId:extra.retryJobId}).then(()=>api.conversations().then(setConversations)).catch(()=>{})}
  const newChat=async()=>{const requestId=++chatOpenSequence.current;try{const created=await api.createConversation();if(requestId!==chatOpenSequence.current)return;conversationIdRef.current=created.id;setConversationId(created.id);setMessages([]);setConversations(rows=>[{id:created.id,title:created.title,created_at:new Date().toISOString(),updated_at:new Date().toISOString()},...rows]);setView("chat")}catch(e){setError(e instanceof Error?e.message:"Could not create a conversation")}}
  const openChat=async(id:string)=>{const requestId=++chatOpenSequence.current;try{const full=await api.getConversation(id);if(requestId!==chatOpenSequence.current)return;conversationIdRef.current=id;setConversationId(id);setMessages(full.messages.map(m=>({id:String(m.id),role:m.role,content:m.content,evidenceCount:m.metadata?.evidenceCount,supplementalSourceCount:m.metadata?.supplementalSourceCount,supplementalSources:m.metadata?.supplementalSources,retryJobId:m.metadata?.retryJobId})));setView("chat")}catch(e){if(requestId===chatOpenSequence.current)setError(e instanceof Error?e.message:"Could not open conversation")}}
  const removeChat=async(id:string)=>{if(!window.confirm("Delete this conversation and its messages?"))return;try{await api.deleteConversation(id);const rows=await api.conversations();setConversations(rows);if(conversationIdRef.current===id){conversationIdRef.current=undefined;setConversationId(undefined);setMessages([]);if(rows[0])await openChat(rows[0].id)}}catch(e){setError(e instanceof Error?e.message:"Could not delete conversation")}}
  const retryJob=async(id:number)=>{
    setRetryingJob(id);setError("")
    try{await api.retryJob(id);await refresh();appendAssistant("I’ve restarted that source from its saved progress.")}
    catch(e){setError(e instanceof Error?e.message:"Could not retry this source")}
    finally{setRetryingJob(null)}
  }
  const createCourse=async()=>{
    const name=window.prompt("Name this course")?.trim()
    if(!name)return
    try{const created=await api.createCourse({name});await refresh();setCourseId(created.id)}catch(e){setError(e instanceof Error?e.message:"Could not create course")}
  }
  const sendMessage=async(text:string,files:File[],retrievalScope:"space_only"|"space_and_related"|"all_sources"="space_only")=>{
    const clean=text.trim()
    if((!clean&&!files.length)||busy)return
    const messageText=[clean,...files.map(file=>`Attached: ${file.name}`)].filter(Boolean).join("\n")
    let activeId=conversationId
    if(!activeId){const created=await api.createConversation();activeId=created.id;conversationIdRef.current=activeId;setConversationId(activeId);setConversations(rows=>[{id:created.id,title:created.title,created_at:new Date().toISOString(),updated_at:new Date().toISOString()},...rows])}
    const nextUser:ChatMessage={id:newId(),role:"user",content:messageText}
    const previousMessages=messages
    setMessages(previous=>[...previous,nextUser]);setBusy(true);setError("");
    try{
      await api.addConversationMessage(activeId,{role:"user",content:messageText})
      if(files.length){
        await api.upload(files,{courseId})
        await refresh()
        appendAssistant(`I’ve added ${files.length===1?"your file":"your files"} to the library. I’ll prepare the content in the background; you can keep chatting while that runs.`)
      }else{
        const link=clean.match(/https?:\/\/(?:www\.)?(?:youtube\.com|youtu\.be)\/[^\s<>]+/i)?.[0]?.replace(/[),.!?]+$/g,"")
        if(link){
          const result=await api.youtube(link,{courseId})
          await refresh()
          if(result.deduplicated&&result.status==="done")appendAssistant("That video is already in your library and ready to study.")
          else if(result.deduplicated&&result.status==="failed")appendAssistant("That video is already in your library, but its last processing attempt failed. I can resume it now.",{retryJobId:result.job_id||undefined})
          else appendAssistant("I’ve added the YouTube source. I’ll fetch its transcript and prepare it for study in the background.")
        }else if(/\b(?:make|create|build)\b.*\b(?:study|learning) plan\b/i.test(clean)){
          const result=await api.syllabus(courseId)
          appendAssistant(result.syllabus||"I couldn’t create a study plan from the current library.")
        }else if(/\b(?:make|create|build)\b.*\bstudy pack\b/i.test(clean)){
          const topic=clean.replace(/^.*?\bstudy pack\b/i,"").replace(/^\s*(?:about|on|for)\s*/i,"").trim()||"the main ideas in my sources"
          const result=await api.studyPack(topic,courseId)
          appendAssistant(result.study_text||"I couldn’t build a study pack from the current library.")
        }else{
          const history:ChatTurn[]=previousMessages.filter(message=>message.content.trim()).slice(-8).map(({role,content})=>({role,content:content.slice(0,3000)}))
          const result=await api.study(clean,courseId,history,retrievalScope)
          appendAssistant(result.answer||"I couldn’t find an answer in your learning library yet.",{evidenceCount:result.sources?.length||0,supplementalSourceCount:result.supplemental_source_count||0,supplementalSources:result.supplemental_sources||[]})
        }
      }
    }catch(e){appendAssistant(`I couldn’t finish that. ${e instanceof Error?e.message:"Please try again."}`)}
    finally{setBusy(false);await refresh()}
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <button className="brand" onClick={()=>setView("plan")}><span className="brand-mark"><Sparkles size={17}/></span><span>LearnOS</span></button>
      <nav className="primary-nav" aria-label="Main navigation">
        <button className={view==="plan"?"selected":""} onClick={()=>setView("plan")}><GraduationCap size={17}/>Study plan</button>
        <button className={view==="library"?"selected":""} onClick={()=>setView("library")}><Library size={17}/>Library <span className="nav-count">{sources.length}</span></button>
        <button className={view==="learn"?"selected":""} onClick={()=>setView("learn")}><Brain size={17}/>Practice</button>
        <button className={view==="chat"?"selected":""} onClick={()=>setView("chat")}><MessageSquare size={17}/>Chat</button>
      </nav>
      <div className="sidebar-chats">
      <Button variant="outline" disabled={busy} onClick={()=>void newChat()}><Plus size={16}/>New chat</Button>
      <div className="sidebar-section"><div className="sidebar-label">Recent chats</div>{conversations.slice(0,12).map(chat=><div className="source-shortcut" key={chat.id}><button className="text-link" disabled={busy} onClick={()=>void openChat(chat.id)} title={chat.title}>{chat.title}</button><button className="icon-button" disabled={busy} aria-label="Delete conversation" onClick={()=>void removeChat(chat.id)}><Trash2 size={13}/></button></div>)}{!conversations.length&&<p className="sidebar-empty">Your conversations will appear here.</p>}</div>
      </div>
      <div className="sidebar-section"><div className="sidebar-label">Your library</div>
        {sources.slice(0,5).map(source=><button className="source-shortcut" key={source.id} onClick={()=>setView("library")} title={source.name}><FileText size={15}/><span>{sourceTitle(source)}</span><i className={`status-dot ${source.status}`} /></button>)}
        {!sources.length&&<p className="sidebar-empty">Sources you add will appear here.</p>}
        {sources.length>5&&<button className="text-link" onClick={()=>setView("library")}>See all {sources.length} sources</button>}
      </div>
      <div className="sidebar-bottom">
        <label className="sidebar-label" htmlFor="course-select">Study space</label>
        <select id="course-select" className="course-select" value={courseId||""} onChange={event=>setCourseId(event.target.value?Number(event.target.value):undefined)}>
          <option value="">All sources</option>{courses.map(course=><option key={course.id} value={course.id}>{course.name}</option>)}
        </select>
        <button className="text-link create-course" onClick={()=>void createCourse()}><Plus size={14}/>Create a course</button>
        {courseId!==undefined&&<button className="text-link delete-space" onClick={()=>setDeletingSpace(courses.find(course=>course.id===courseId)||null)}><Trash2 size={14}/>Delete this space</button>}
        <div className="connection"><i className={health?.status==="ok"?"online":"offline"}/>{health?.status==="ok"?"Connected":"Waiting for server"}</div>
      </div>
    </aside>

    <main className="main-area">
      <header className="app-header">
        <div className="header-title"><div className="mobile-brand"><span className="brand-mark"><Sparkles size={16}/></span>LearnOS</div><h1>{view==="chat"?"Chat":view==="plan"?"Study plan":view==="learn"?"Practice":"Your library"}</h1><div className="mobile-course-control"><select aria-label="Choose course" className="course-select" value={courseId||""} onChange={event=>setCourseId(event.target.value?Number(event.target.value):undefined)}><option value="">All sources</option>{courses.map(course=><option key={course.id} value={course.id}>{course.name}</option>)}</select><button className="text-link" onClick={()=>void createCourse()}><Plus size={14}/>Course</button></div></div>
        <div className="header-actions">
          {activeJobs.length>0&&<Badge><LoaderCircle size={13}/>{activeJobs.length} processing</Badge>}
          <button className="icon-button" aria-label="Refresh" title="Refresh" onClick={()=>{void refresh();if(view==="plan")setPlanRegenerateKey(key=>key+1)}}><RefreshCw size={16}/></button>
          <Button onClick={()=>setAddOpen(true)}><Plus size={16}/><span>Add source</span></Button>
        </div>
      </header>
      {error&&<div className="global-error" role="alert">{error}</div>}
      {view==="chat"?<ChatView messages={messages} busy={busy} activeJobs={activeJobs} failedJobs={failedJobs} retryingJob={retryingJob} retry={retryJob} send={sendMessage} addSource={()=>setAddOpen(true)} courseSelected={courseId!==undefined} openPlan={()=>setView("plan")}/>:view==="learn"?<LearnView courseId={courseId} sourceCount={sources.length} readyCount={sources.filter(source=>source.status==="done").length} openLibrary={()=>setView("library")} aiReady={health?.llm===true} addSource={()=>setAddOpen(true)} openPlan={()=>setView("plan")}/>:view==="plan"?<StudyPlanView courseId={courseId} courseName={courses.find(course=>course.id===courseId)?.name} sources={sources} sourcesLoaded={health!==undefined&&sourcesFor===(courseId===undefined?"all":String(courseId))} aiReady={health?.llm===true} refreshKey={planRefreshKey} regenerateKey={planRegenerateKey} rebuild={()=>setPlanRegenerateKey(key=>key+1)} addSource={()=>setAddOpen(true)} openLibrary={()=>setView("library")}/>:<LibraryView aiReady={health?.llm===true} courseName={courses.find(course=>course.id===courseId)?.name} sources={sources} courses={courses} jobs={currentJobs} failedJobs={failedJobs} retryingJob={retryingJob} retry={retryJob} refresh={refresh} addSource={()=>setAddOpen(true)} deleteSpace={courseId!==undefined?()=>setDeletingSpace(courses.find(course=>course.id===courseId)||null):undefined}/>}
    </main>
    {addOpen&&<AddSourceDialog courseId={courseId} courses={courses} close={()=>setAddOpen(false)} refresh={refresh}/>}
    {deletingSpace&&<DeleteSpaceDialog course={deletingSpace} close={()=>setDeletingSpace(null)} done={()=>{setDeletingSpace(null);if(courseId===deletingSpace.id)setCourseId(undefined);else void refresh()}}/>}
  </div>
}

// The study plan page shows one thing to do at a time: add material, wait while LearnOS prepares it, fix a source
// that needs attention, or study: copy the session for an AI tutor, then mark it done and go on.
function StudyPlanView({courseId,courseName,sources,sourcesLoaded,aiReady,refreshKey,regenerateKey,rebuild,addSource,openLibrary}:{courseId?:number;courseName?:string;sources:Source[];sourcesLoaded:boolean;aiReady:boolean;refreshKey:number;regenerateKey:number;rebuild:()=>void;addSource:()=>void;openLibrary:()=>void}){
  const handledRegenerate=useRef(regenerateKey)
  const[syllabus,setSyllabus]=useState("")
  const[planWarnings,setPlanWarnings]=useState<string[]>([])
  const[relationshipInfo,setRelationshipInfo]=useState<PlanRelationships>({sources:[],cross_paths:[],shared_foundations:[]})
  const[topic,setTopic]=useState("")
  const[topicId,setTopicId]=useState<string>()
  const[planNodes,setPlanNodes]=useState<OutlineTopic[]>([])
  const[tutor,setTutorState]=useState(()=>remembered("learnos.tutor",["general","chatgpt","gemini","claude"],"general"))
  const setTutor=(value:string)=>{setTutorState(value);remember("learnos.tutor",value)}
  const[maxInputTokens,setMaxInputTokens]=useState(6000)
  const[pack,setPack]=useState<StudyPackExport|null>(null)
  const[partlyIds,setPartlyIds]=useState<Set<string>>(new Set())
  const[completedTopics,setCompletedTopics]=useState<Set<string>>(new Set())
  const[busy,setBusy]=useState(false)
  const[error,setError]=useState("")
  const[planStatus,setPlanStatus]=useState<PlanStatus>("ready")
  const[planError,setPlanError]=useState("")
  const[generatedAt,setGeneratedAt]=useState<string|null>(null)
  const[completedIds,setCompletedIds]=useState<Set<string>>(new Set())
  const[pollTick,setPollTick]=useState(0)
  const[reloadKey,setReloadKey]=useState(0)
  const[loadedFor,setLoadedFor]=useState<string|null>(null)  // the space whose plan is shown: nothing is judged before it loads
  const[sessionSize,setSessionSizeState]=useState<SessionSize>(()=>remembered("learnos.session-size",["compact","standard","large","xl"],"standard") as SessionSize)
  const setSessionSize=(value:SessionSize)=>{setSessionSizeState(value);remember("learnos.session-size",value)}
  const[partlyBudgets,setPartlyBudgets]=useState<Record<string,number>>({})
  // The open plan topic, then each part opened inside it; the last entry is shown.
  const[unitTrail,setUnitTrail]=useState<PreparedTopic[]>([])
  const[unitBusy,setUnitBusy]=useState(false)
  const[unitError,setUnitError]=useState("")
  const unitRequest=useRef(0)
  const unitTarget=useRef<{nodeId:string;trail:PreparedTopic[]}|null>(null)
  const doneIds=useRef(completedIds);doneIds.current=completedIds
  const[packError,setPackError]=useState("")
  const[masterPrompt,setMasterPrompt]=useState<MasterPrompt|null>(null)
  const[masterError,setMasterError]=useState("")
  const[masterTick,setMasterTick]=useState(0)  // a saved tick: the master prompt is read again after the save, not before
  const activeCourse=useRef(courseId)
  const scopeKey=courseId===undefined?"all":String(courseId)
  const applyPlan=(plan:Awaited<ReturnType<typeof api.syllabus>>,progress:Awaited<ReturnType<typeof api.studyProgress>>,requested:number|undefined)=>{
    setSyllabus(plan.syllabus||"");setPlanWarnings(plan.warnings||[]);setRelationshipInfo(normalizePlanRelationships(plan.source_relationships))
    setPlanStatus(plan.status||"ready");setPlanError(plan.error||"");setGeneratedAt(plan.generated_at||null)
    setPlanNodes((plan.nodes||[]).map(node=>({id:node.id,parentId:node.parent_id||"",pathId:node.path_id||"main",kind:node.type,topic:node.title,purpose:node.metadata?.purpose||"",prerequisites:node.metadata?.prerequisites||"",groups:Array.isArray(node.metadata?.source_groups)?node.metadata.source_groups:String(node.metadata?.source_groups||"").split(",").map(s=>s.trim()).filter(Boolean),why:node.metadata?.why_included||"",enables:node.metadata?.enables||"",sourceIds:Array.isArray(node.metadata?.source_ids)?node.metadata.source_ids:[],sourceRoles:Array.isArray(node.metadata?.source_roles)?node.metadata.source_roles:[],confidence:node.metadata?.topic_merge_confidence||0})))
    applyProgress(progress)
    setLoadedFor(requested===undefined?"all":String(requested))
  }
  const applyProgress=(progress:Awaited<ReturnType<typeof api.studyProgress>>)=>{setCompletedTopics(new Set((progress.completed_legacy_topics??progress.completed_topics).map(normalizedTopic)));setCompletedIds(new Set(progress.completed_topic_ids||[]));setPartlyIds(new Set(progress.partly_done_topic_ids||[]));setPartlyBudgets(progress.partly_done_budgets||{})}
  useEffect(()=>{let current=true;const regenerate=regenerateKey!==handledRegenerate.current;handledRegenerate.current=regenerateKey;if(activeCourse.current!==courseId||regenerate){unitRequest.current+=1;setUnitTrail([]);setUnitBusy(false);setUnitError("")}if(activeCourse.current!==courseId){setPlanNodes([]);setPlanStatus("ready");setPlanError("");setGeneratedAt(null);setCompletedIds(new Set());setCompletedTopics(new Set());setPartlyIds(new Set());setMasterPrompt(null);setLoadedFor(null)}activeCourse.current=courseId;setPack(null);setTopicId(undefined);setTopic("");setPackError("");setBusy(true);setError("");const requested=courseId;Promise.all([api.syllabus(requested,regenerate),api.studyProgress(requested)]).then(([plan,progress])=>{if(current)applyPlan(plan,progress,requested)}).catch(e=>{if(current)setError(e instanceof Error?e.message:"Could not load your study plan")}).finally(()=>{if(current)setBusy(false)});return()=>{current=false}},[courseId,refreshKey,regenerateKey,reloadKey])
  const progressKey=[...completedIds].sort().join("|")+"#"+[...completedTopics].sort().join("|")
  useEffect(()=>{if(!planNodes.length){setMasterPrompt(null);return}let current=true;const requested=courseId;api.masterPrompt(requested,tutor).then(result=>{if(current&&activeCourse.current===requested){setMasterPrompt(result);setMasterError("")}}).catch(e=>{if(current)setMasterError(e instanceof Error?e.message:"Could not prepare the master prompt")});return()=>{current=false}},[courseId,tutor,generatedAt,planNodes.length,progressKey,masterTick])
  // While a plan is being built in the background, check again every few seconds; reading never builds.
  useEffect(()=>{if(planStatus!=="updating"&&planStatus!=="outdated")return;const requested=courseId;const timer=window.setTimeout(()=>{Promise.all([api.syllabus(requested),api.studyProgress(requested)]).then(([plan,progress])=>{if(activeCourse.current===requested){applyPlan(plan,progress,requested);setPollTick(tick=>tick+1)}}).catch(()=>{if(activeCourse.current===requested)setPollTick(tick=>tick+1)})},4000);return()=>window.clearTimeout(timer)},[planStatus,pollTick,courseId])
  const makePack=async(event:FormEvent)=>{event.preventDefault();if(!topic.trim())return;setBusy(true);setPackError("");try{const result=await api.exportStudyPack(topic.trim(),courseId,tutor,maxInputTokens,topicId);setPack(result)}catch(e){setPackError(e instanceof Error?e.message:"Could not prepare study pack")}finally{setBusy(false)}}
  // Opens a plan topic (trail empty) or a part of the last topic in `trail`. retry=false only shows it again.
  // A topic too large for one session opens at its first unfinished part, unless its list of parts was asked for.
  const openUnit=async(nodeId:string,trail:PreparedTopic[],options:{size?:SessionSize;tutor?:string;retry?:boolean;showParts?:boolean}={})=>{
    const size=options.size??sessionSize,tutorName=options.tutor??tutor
    if(unitBusy&&unitTarget.current?.nodeId===nodeId&&!options.size&&!options.tutor)return  // a second click on what is loading
    const request=++unitRequest.current;const requested=courseId;unitTarget.current={nodeId,trail};setUnitBusy(true);setUnitError("")
    try{
      const result=await api.prepareTopic(nodeId,requested,size,tutorName,options.retry??true)
      if(request!==unitRequest.current||activeCourse.current!==requested)return
      const nextTrail=[...trail,result];setUnitTrail(nextTrail)
      const firstOpen=result.status==="split"&&!options.showParts?(result.children||[]).find(part=>!doneIds.current.has(part.id)):undefined
      if(firstOpen)void openUnit(firstOpen.id,nextTrail,{size,tutor:tutorName,retry:options.retry})
    }catch(e){
      if(request!==unitRequest.current)return
      const status=(e as {status?:number}).status
      if(trail.length&&(status===404||status===409)){void openUnit(trail[0].node_id,[],{size,tutor:tutorName});setUnitError("Your study plan changed, so this topic was prepared again.");return}
      if(status===404)setUnitTrail([])
      setUnitError(status===404?"This topic is no longer in your study plan.":e instanceof Error?e.message:"Could not prepare this topic")
    }finally{if(request===unitRequest.current)setUnitBusy(false)}}
  const currentUnit=unitTrail.length?unitTrail[unitTrail.length-1]:null
  const pendingUnit=unitBusy?unitTarget.current:null
  // Parts are made for one session size, so a new size starts again from the plan topic.
  const changeSize=(size:SessionSize)=>{setSessionSize(size);const root=pendingUnit?(pendingUnit.trail[0]?.node_id??pendingUnit.nodeId):unitTrail[0]?.node_id;/* the latest choice wins */setUnitTrail([]);if(root)void openUnit(root,[],{size})}
  // The tutor only changes a name in the text; a topic divided into parts has no text.
  // "About you" changed: what is added to the open unit and the master prompt changes with it.
  // Called after the save returns, through a ref, so it sees what is open (or still loading) by then.
  const contextChanged=()=>{setMasterTick(tick=>tick+1);if(pendingUnit){void openUnit(pendingUnit.nodeId,pendingUnit.trail,{tutor,retry:false});return}if(currentUnit&&currentUnit.status!=="split")void openUnit(currentUnit.node_id,unitTrail.slice(0,-1),{retry:false})}
  const contextChangedRef=useRef(contextChanged);contextChangedRef.current=contextChanged
  const changeTutor=(name:string)=>{setTutor(name);if(pendingUnit){void openUnit(pendingUnit.nodeId,pendingUnit.trail,{tutor:name,retry:false});return}if(currentUnit&&currentUnit.status!=="split")void openUnit(currentUnit.node_id,unitTrail.slice(0,-1),{tutor:name,retry:false})}
  const outline=planNodes
  const topicOptions=outline.filter(item=>item.kind==="study").map(item=>item.topic)
  const isDone=(item:OutlineTopic)=>completedIds.has(item.id)||completedTopics.has(normalizedTopic(item.topic))
  const markDone=async(id:string,title:string,completed:boolean,showError:(message:string)=>void=setUnitError)=>{const isPlanTopic=planNodes.some(node=>node.id===id);const next=new Set(completedTopics);const nextIds=new Set(completedIds);if(completed){nextIds.add(id)}else{if(isPlanTopic)next.delete(normalizedTopic(title));nextIds.delete(id)}setCompletedTopics(next);setCompletedIds(nextIds);const requested=courseId;try{await api.updateStudyProgress(title,requested,completed,id)}catch(e){setCompletedTopics(completedTopics);setCompletedIds(completedIds);showError(e instanceof Error?e.message:"Could not save your progress");return false}
    // Finishing the last part of a topic finishes the topic too (and un-ticking a part re-opens it): read it back.
    setMasterTick(tick=>tick+1);api.studyProgress(requested).then(progress=>{if(activeCourse.current===requested)applyProgress(progress)}).catch(()=>{});return true}
  const scrollToStudy=()=>window.setTimeout(()=>document.getElementById("study-card")?.scrollIntoView({behavior:"smooth",block:"start"}),0)
  // The next unit shares the open trail down to its own depth: a sibling part keeps the parent, the next topic starts fresh.
  const sizeFor=(id:string):SessionSize|undefined=>{const name=Object.entries(SIZE_TOKENS).find(([,tokens])=>tokens===partlyBudgets[id])?.[0] as SessionSize|undefined;return name&&name!==sessionSize?name:undefined}
  const openTopic=(id:string)=>{const size=sizeFor(id);if(size)setSessionSize(size);void openUnit(id,[],size?{size}:{})}
  const openNext=(next:{id:string;depth:number})=>{if(next.depth===0)openTopic(next.id);else void openUnit(next.id,unitTrail.slice(0,next.depth));scrollToStudy()}
  const doneAndNext=async()=>{if(!currentUnit)return;const course=courseId,request=unitRequest.current
    if(!unitDone&&!(await markDone(currentUnit.node_id,currentUnit.title,true)))return
    if(activeCourse.current!==course||unitRequest.current!==request)return  // the learner moved on while it was saving
    if(currentUnit.next)openNext(currentUnit.next)}
  const toggleComplete=(item:OutlineTopic,completed:boolean)=>markDone(item.id,item.topic,completed,setError)
  const unitDone=currentUnit?completedIds.has(currentUnit.node_id)||(currentUnit.depth===0&&completedTopics.has(normalizedTopic(currentUnit.title))):false
  const selectTopic=(item:OutlineTopic)=>{setTopic(item.topic);setTopicId(item.id);setPack(null);openTopic(item.id);scrollToStudy()}
  const builtAt=generatedAt?new Date(generatedAt.endsWith("Z")||generatedAt.includes("+")?generatedAt:generatedAt+"Z").toLocaleString():""
  const statusLine=planStatus==="updating"?"Updating with your latest material. You can keep studying meanwhile.":planStatus==="failed"?planError:planStatus==="outdated"?"Your material changed; the plan updates by itself shortly.":builtAt?`Up to date · built ${builtAt}`:"Up to date"
  // What the page is about right now.
  const studyItems=outline.filter(item=>item.kind==="study")
  const studiedCount=studyItems.filter(isDone).length
  const readyCount=sources.filter(source=>source.status==="done").length
  const needsInputCount=sources.filter(source=>source.status==="needs_input").length
  const attentionCount=needsInputCount+sources.filter(source=>source.status==="failed").length
  const processingCount=sources.length-readyCount-attentionCount
  const loaded=loadedFor===scopeKey
  const stage=!loaded?(error?"error":"loading"):!studyItems.length&&!sourcesLoaded?"loading"
    :studyItems.length?"ready"
    :!sources.length?"empty"
    :!readyCount&&attentionCount&&!processingCount?"attention"
    :planStatus==="failed"?"failed"
    :processingCount||!readyCount||planStatus==="updating"||planStatus==="outdated"?"preparing"
    :!aiReady?"no-ai":"no-topics"
  // The next topic: from the plan's reading order once known, else the first unstudied topic.
  const nextTopic=masterPrompt?masterPrompt.next_topic:(()=>{const item=studyItems.find(entry=>!isDone(entry));return item?{id:item.id,title:item.topic,partly_done:partlyIds.has(item.id)}:null})()
  const nextNode=nextTopic?outline.find(item=>item.id===nextTopic.id):undefined
  const started=Boolean(masterPrompt?.started||studiedCount)
  const firstOpenPart=currentUnit?.status==="split"?(currentUnit.children||[]).find(part=>!completedIds.has(part.id)):undefined
  const nextLabel=currentUnit?.next?(currentUnit.next.depth>0?"Done · next part":"Done · next topic"):""
  // At the end of a learning path the plan may still have topics elsewhere (another path, or one skipped).
  const planNext=nextTopic&&nextTopic.id!==unitTrail[0]?.node_id?nextTopic:null
  const finishAndGo=async()=>{if(!currentUnit||!planNext)return;const course=courseId;if(!unitDone&&!(await markDone(currentUnit.node_id,currentUnit.title,true)))return;if(activeCourse.current===course)openNext({id:planNext.id,depth:0})}
  const message=(title:string,text:ReactNode,actions?:ReactNode,spinner=false)=><div className="plan-message"><h2>{spinner&&<LoaderCircle size={17} className="learn-spin"/>}{title}</h2><p>{text}</p>{actions&&<div className="plan-actions">{actions}</div>}</div>
  return <section className="plan-view">
    {stage==="loading"?message("Loading your study plan…","",undefined,true)
    :stage==="error"?message("Your study plan could not be loaded",error,<Button variant="outline" onClick={()=>setReloadKey(key=>key+1)}><RefreshCw size={14}/>Try again</Button>)
    :stage==="empty"?message(courseName?`Add material to ${courseName}`:"Add your learning material","YouTube videos or playlists, PDFs, slides, documents or your own notes. LearnOS reads them, builds your study plan and prepares each study session for you.",<Button onClick={addSource}><Plus size={16}/>Add material</Button>)
    :stage==="attention"?message("Your material needs your attention",needsInputCount===attentionCount?`${plural(attentionCount,"source")} had no text LearnOS could read, for example a video without captions. You can paste its text or transcript in the Library.`:`Processing stopped for ${plural(attentionCount,"source")}. In the Library you can resume it, or paste its text.`,<><Button onClick={openLibrary}>Open the Library</Button><Button variant="ghost" onClick={addSource}><Plus size={14}/>Add other material</Button></>)
    :stage==="preparing"?message("Preparing your study plan",<>LearnOS is reading your material and building the plan{processingCount?` (${plural(processingCount,"source")} still being read)`:""}. This page updates by itself; you can leave and come back.</>,undefined,true)
    :stage==="failed"?message("The study plan could not be built",planError||"The AI service did not answer.",<Button variant="outline" onClick={rebuild}><RefreshCw size={14}/>Try again</Button>)
    :stage==="no-ai"?message("Study plans need an AI model","LearnOS needs a Gemini API key to find the topics in your material. Add GEMINI_API_KEY to the .env file and restart LearnOS; your material is kept.")
    :stage==="no-topics"?message("No topics found yet","LearnOS could not find study topics in this material yet. This is usually a temporary problem with the AI service.",<Button variant="outline" onClick={rebuild}><RefreshCw size={14}/>Try again</Button>)
    :<>
      {attentionCount>0&&<p className="plan-warning" role="status">{plural(attentionCount,"source")} could not be read and {attentionCount===1?"is":"are"} not in this plan yet. <button className="link-button" onClick={openLibrary}>Fix it in the Library</button></p>}
      <div className="plan-section study-card" id="study-card">
        <div className="study-progress"><span>{courseName||"All sources"} · {studiedCount} of {plural(studyItems.length,"topic")} studied</span><progress className="plan-progress" max={studyItems.length} value={studiedCount} aria-label="Topics studied"/></div>
        {unitError&&<p className="plan-error" role="alert">{unitError}</p>}
        {unitBusy?<div className="study-busy" role="status"><LoaderCircle size={16} className="learn-spin"/>Preparing your study session…</div>
        :currentUnit&&currentUnit.status!=="split"?<>
          {unitTrail.length>1&&<nav className="unit-trail" aria-label="Topic and its parts">{unitTrail.map((entry,index)=>index<unitTrail.length-1?<span key={entry.node_id}><button type="button" className="link-button" onClick={()=>void openUnit(entry.node_id,unitTrail.slice(0,index),{retry:false,showParts:true})}>{entry.title}</button><span aria-hidden="true"> › </span></span>:<strong key={entry.node_id}>{entry.title}</strong>)}</nav>}
          <p className="eyebrow">{unitDone?"Studied":"Now studying"}</p>
          <h2 className="study-title">{currentUnit.title}</h2>
          {currentUnit.purpose&&<p className="plan-help">{currentUnit.purpose}</p>}
          {currentUnit.warning&&<p className="plan-warning" role="status">{currentUnit.warning}</p>}
          {currentUnit.note&&<p className="plan-warning" role="status">{currentUnit.note}</p>}
          {currentUnit.overlaps&&currentUnit.overlaps.length>0&&<p className="plan-help overlap-note">Shares key ideas with {currentUnit.overlaps.map(item=>`“${item.title}”`).join(" and ")}: {Array.from(new Set(currentUnit.overlaps.flatMap(item=>item.shared))).slice(0,4).join(", ")}. Your tutor recaps those briefly and spends the session on what is new{currentUnit.new_ideas&&currentUnit.new_ideas.length?` (${currentUnit.new_ideas.slice(0,4).join(", ")})`:""}.</p>}
          <div className="study-actions"><CopyAnswer text={currentUnit.study_text||""} label="Copy for your AI tutor" primary/>{currentUnit.next?<Button onClick={()=>void doneAndNext()}>{unitDone?(nextLabel.replace("Done · ","Go to ")):nextLabel}</Button>:planNext?<Button onClick={()=>void finishAndGo()}>{unitDone?"Next":"Done · next"}: {planNext.title}</Button>:<Button variant="outline" disabled={unitDone} onClick={()=>void markDone(currentUnit.node_id,currentUnit.title,true)}>{unitDone?"Studied":"Mark as studied"}</Button>}</div>
          <p className="plan-help">Paste it into ChatGPT, Claude or Gemini and study there. When you finish, write “unit done” in that chat for a recap and review questions.{currentUnit.next?` Next: ${currentUnit.next.title}.`:planNext?"":unitDone?" You have studied every topic in this plan.":currentUnit.depth?" This is the last part of this learning path.":" This is the last topic of this learning path."}</p>
          <div className="study-secondary"><Button variant="ghost" onClick={()=>downloadText(`${fileSlug(currentUnit.title)}.md`,currentUnit.study_text||"")}><Download size={14}/>Download</Button>{unitDone&&<button className="text-link" onClick={()=>void markDone(currentUnit.node_id,currentUnit.title,false)}>Mark as not studied</button>}</div>
          <details className="outline-markdown"><summary>Preview what is copied (about {(currentUnit.estimated_tokens||0).toLocaleString()} tokens, {plural((currentUnit.sources||[]).length,"passage")} from your material)</summary><div className="plan-syllabus"><MarkdownText text={currentUnit.study_text||""}/></div></details>
        </>
        :currentUnit?<>
          <p className="eyebrow">A topic in parts</p>
          <h2 className="study-title">{currentUnit.title}</h2>
          <p className="plan-help">{currentUnit.reason} Each part is one study session.</p>
          <div className="outline-topic-list unit-parts">{(currentUnit.children||[]).map(part=>{const done=completedIds.has(part.id);return <article className={`outline-topic${done?" completed":""}`} key={part.id}><label className="outline-check"><input type="checkbox" checked={done} onChange={event=>void markDone(part.id,part.title,event.target.checked)} aria-label={`Mark ${part.title} ${done?"not studied":"studied"}`}/><span>{done?<Check size={13}/>:null}</span></label><button className="outline-topic-button" onClick={()=>void openUnit(part.id,unitTrail)}><strong>{part.title}</strong>{part.purpose&&<small>{part.purpose}</small>}</button></article>})}</div>
          <div className="study-actions">{firstOpenPart?<Button onClick={()=>void openUnit(firstOpenPart.id,unitTrail)}>Continue with: {firstOpenPart.title}</Button>:currentUnit.next?<Button onClick={()=>openNext(currentUnit.next!)}>Next: {currentUnit.next.title}</Button>:planNext?<Button onClick={()=>openNext({id:planNext.id,depth:0})}>Next: {planNext.title}</Button>:<span className="plan-help">All parts studied. You have studied every topic in this plan.</span>}</div>
        </>
        :nextTopic?<>
          <p className="eyebrow">{started?"Up next":"Start here"}</p>
          <h2 className="study-title">{nextTopic.title}{nextTopic.partly_done?<span className="tree-badge partly">Partly done</span>:null}</h2>
          {nextNode?.purpose&&<p className="plan-help">{nextNode.purpose}</p>}
          <div className="study-actions"><Button onClick={()=>openNext({id:nextTopic.id,depth:0})}>{started?"Continue studying":"Start studying"}</Button></div>
          <p className="plan-help">LearnOS prepares the study session for you. You copy it into your AI tutor and study.</p>
        </>
        :<><p className="eyebrow">All done</p><h2 className="study-title">You have studied every topic</h2><p className="plan-help">Choose any topic below to review it, or add more material.</p></>}
        <div className="session-settings"><label>Session length<select value={sessionSize} onChange={event=>changeSize(event.target.value as SessionSize)}><option value="compact">Short (about 4k tokens)</option><option value="standard">Standard (about 8k tokens)</option><option value="large">Long (about 16k tokens)</option><option value="xl">Very long (about 32k tokens)</option></select></label><label>Written for<select value={tutor} onChange={event=>changeTutor(event.target.value)}><option value="general">Any AI tutor</option><option value="chatgpt">ChatGPT</option><option value="gemini">Gemini</option><option value="claude">Claude</option></select></label></div>
      </div>
      <div className="plan-section" id="plan-outline"><div className="plan-section-heading"><div><h3>Your study plan</h3><span className="plan-help">Choose any topic to study it. Groups open and close.</span></div></div>
        <p className={`plan-status ${planStatus}`} role="status">{planStatus==="updating"&&<LoaderCircle size={13} className="learn-spin"/>}{statusLine}</p>
        {error&&<p className="plan-error" role="alert">{error}</p>}
        {planWarnings.length>0&&<div className="plan-warning" role="status"><strong>Part of this plan is missing.</strong><ul>{planWarnings.map((warning,index)=><li key={index}>{warning}</li>)}</ul><span>This is usually a temporary problem with the AI service. <button className="link-button" onClick={rebuild}>Try again</button></span></div>}
        <OutlineTree nodes={outline} isDone={isDone} partly={partlyIds} nextId={nextTopic?.id} open={selectTopic} toggle={(item,completed)=>void toggleComplete(item,completed)}/>
        {(relationshipInfo.sources.length>0||relationshipInfo.shared_foundations.length>0)&&<details className="outline-markdown"><summary>How LearnOS grouped your sources</summary><div className="plan-help">Source identity is inferred and may be uncertain. Shared topics alone do not establish that sources belong to one course.</div>{relationshipInfo.sources.map((r:any,index:number)=><p key={`${r.source_a_id}-${r.source_b_id}-${index}`}>Source {r.source_a_id} + source {r.source_b_id}: <strong>{String(r.relation_type||"").replaceAll("_"," ")}</strong> ({Math.round(Number(r.confidence||0)*100)}% confidence). {r.rationale}</p>)}{relationshipInfo.shared_foundations.length>0&&<p>Shared foundations: {relationshipInfo.shared_foundations.join(", ")}</p>}</details>}
        <details className="outline-markdown"><summary>The whole plan as text</summary><div className="plan-actions"><CopyAnswer text={syllabus}/></div><div className="plan-syllabus"><MarkdownText text={syllabus}/></div></details>
      </div>
      <details className="plan-section optional-section"><summary><strong>Tell your tutor about you</strong><span>Optional. Your background, goal and how you like to practise are added to every study session you copy, so the tutor pitches it right.</span></summary><AboutYou saved={()=>contextChangedRef.current()}/></details>
      <details className="plan-section optional-section"><summary><strong>Using a ChatGPT or Claude Project, or a Gemini Gem?</strong><span>Optional. Each study session already works on its own; this gives your tutor the whole plan once.</span></summary>
        {masterError&&<p className="plan-error" role="alert">{masterError}</p>}
        {masterPrompt&&<><p className="plan-help">{TUTOR_SETUP[tutor]||TUTOR_SETUP.general} It lists your whole plan and your progress ({masterPrompt.completed} of {plural(masterPrompt.topics,"topic")} studied), so update it now and then.</p>
        <div className="plan-actions"><CopyAnswer text={masterPrompt.text}/><Button variant="ghost" onClick={()=>downloadText(`${fileSlug(`study plan ${courseName||"all sources"}`)}.md`,masterPrompt.text)}><Download size={14}/>Download</Button></div>
        <details className="outline-markdown"><summary>Preview (about {masterPrompt.estimated_tokens.toLocaleString()} tokens)</summary><pre className="master-preview">{masterPrompt.text}</pre></details></>}
      </details>
      <details className="plan-section optional-section"><summary><strong>Study something not in the plan</strong><span>Optional. LearnOS searches all your passages for any topic you type.</span></summary><form className="pack-form" onSubmit={event=>void makePack(event)}><label className="field-label">Topic<input className="text-input" list="syllabus-topics" value={topic} onChange={event=>{setTopic(event.target.value);setTopicId(undefined)}} placeholder="Type any topic" required/><datalist id="syllabus-topics">{topicOptions.map((value,index)=><option key={`${value}-${index}`} value={value}/>)}</datalist></label><label className="field-label">Size<select className="text-input" value={maxInputTokens} onChange={event=>setMaxInputTokens(Number(event.target.value))}><option value={4000}>Short · about 4k tokens</option><option value={6000}>Standard · about 6k tokens</option><option value={10000}>Long · about 10k tokens</option></select></label><Button type="submit" disabled={busy||!topic.trim()}>{busy?<><LoaderCircle size={15}/>Preparing…</>:<><Sparkles size={15}/>Prepare</>}</Button></form>
        {packError&&<p className="plan-error" role="alert">{packError}</p>}{pack&&<div className="study-pack-result"><div className="plan-section-heading"><h3>Ready to copy</h3><CopyAnswer text={pack.study_text}/></div><p className="plan-help">About {pack.estimated_tokens.toLocaleString()} tokens · {pack.sources.length} passages{pack.omitted_passages?` · ${pack.omitted_passages} more passages did not fit`:""}.</p>{pack.omitted_passages>0&&<div className="pack-subtopics"><strong>This topic is broader than one session.</strong><p>Narrow it to one of these related topics, or choose a longer size.</p>{pack.suggested_subtopics.map((value,index)=><button type="button" key={`${value}-${index}`} onClick={()=>{setTopic(value);setTopicId(undefined);setPack(null)}}>{value}</button>)}</div>}<div className="plan-syllabus"><MarkdownText text={pack.study_text}/></div></div>}
      </details>
    </>}
  </section>
}

// The plan's outline as a tree: groups open and close in place, each topic shows whether it is studied or next.
// Large plans open only the branch with the next topic; a branch never closes by itself once opened.
function OutlineTree({nodes,isDone,partly,nextId,open,toggle}:{nodes:OutlineTopic[];isDone:(item:OutlineTopic)=>boolean;partly:Set<string>;nextId?:string;open:(item:OutlineTopic)=>void;toggle:(item:OutlineTopic,completed:boolean)=>void}){
  const ids=new Set(nodes.map(node=>node.id))
  const children=new Map<string,OutlineTopic[]>()
  const roots:OutlineTopic[]=[]
  nodes.forEach(node=>{if(!node.parentId||node.parentId.toLowerCase()==="none"||!ids.has(node.parentId)||node.parentId===node.id)roots.push(node);else children.set(node.parentId,[...(children.get(node.parentId)||[]),node])})
  // Nodes a malformed plan leaves unreachable (parents that loop) are shown at the end rather than lost.
  const reached=new Set<string>();const visit=(id:string)=>{if(reached.has(id))return;reached.add(id);(children.get(id)||[]).forEach(child=>visit(child.id))};roots.forEach(root=>visit(root.id))
  nodes.forEach(node=>{if(!reached.has(node.id)){roots.push(node);visit(node.id)}})
  const below=new Map<string,{done:number;total:number}>()
  const count=(id:string,seen:Set<string>):{done:number;total:number}=>{const cached=below.get(id);if(cached)return cached;if(seen.has(id))return {done:0,total:0};seen.add(id);let done=0,total=0;for(const child of children.get(id)||[]){if(child.kind==="study"){total++;if(isDone(child))done++}const inner=count(child.id,seen);done+=inner.done;total+=inner.total}const result={done,total};below.set(id,result);return result}
  const parentOf=new Map(nodes.map(node=>[node.id,node.parentId]))
  const everOnPath=useRef(new Set<string>())  // branches opened for "Up next" stay open when it moves on
  let cursor=nextId?parentOf.get(nextId):undefined;for(let guard=0;cursor&&guard<20;guard++){everOnPath.current.add(cursor);cursor=parentOf.get(cursor)}
  const small=nodes.filter(node=>node.kind==="study").length<=40
  const[expanded,setExpanded]=useState<Record<string,boolean>>({})
  const isOpen=(id:string)=>expanded[id]??(small||everOnPath.current.has(id))
  const flip=(id:string)=>setExpanded(value=>({...value,[id]:!isOpen(id)}))
  const row=(item:OutlineTopic,depth:number,seen:Set<string>):ReactNode=>{
    if(seen.has(item.id)||depth>12)return null
    const inner=new Set(seen).add(item.id)
    const kids=children.get(item.id)||[]
    if(item.kind==="study"){const done=isDone(item);const next=item.id===nextId
      return <div key={item.id}><div className={`tree-row${done?" completed":""}${next?" next":""}`}>
        <label className="outline-check"><input type="checkbox" checked={done} onChange={event=>toggle(item,event.target.checked)} aria-label={`Mark ${item.topic} ${done?"not studied":"studied"}`}/><span>{done?<Check size={13}/>:null}</span></label>
        <button className="outline-topic-button" onClick={()=>open(item)}><strong>{item.topic}</strong>{item.purpose&&<small>{item.purpose}</small>}{item.prerequisites&&item.prerequisites!=="None recorded"&&item.prerequisites.toLowerCase()!=="none"&&<em>Before this: {item.prerequisites}</em>}</button>
        <span className="tree-badges">{next&&<span className="tree-badge next">Up next</span>}{!done&&partly.has(item.id)&&<span className="tree-badge partly">Partly done</span>}</span>
      </div>{kids.length>0&&<div className="tree-children">{kids.map(child=>row(child,depth+1,inner))}</div>}</div>}
    const isExpanded=isOpen(item.id);const {done,total}=count(item.id,new Set())
    return <div key={item.id} className="tree-group">
      <button className="tree-toggle" aria-expanded={isExpanded} onClick={()=>flip(item.id)}>{isExpanded?<ChevronDown size={15}/>:<ChevronRight size={15}/>}<span><strong>{item.topic}</strong><small>{total?`${done} of ${plural(total,"topic")} studied`:"No topics yet"}</small></span></button>
      {isExpanded&&<div className="tree-children">{kids.map(child=>row(child,depth+1,inner))}</div>}
    </div>
  }
  return <div className="outline-tree">{roots.map(root=>{
    if(root.kind==="study")return row(root,0,new Set())
    const isExpanded=isOpen(root.id);const {done,total}=count(root.id,new Set())
    return <section className="outline-group" key={root.id}><button className="tree-root" aria-expanded={isExpanded} onClick={()=>flip(root.id)}>{isExpanded?<ChevronDown size={16}/>:<ChevronRight size={16}/>}<span>{root.topic}</span><small>{done} of {plural(total,"topic")} studied</small></button>{isExpanded&&<div className="outline-topic-list">{(children.get(root.id)||[]).map(child=>row(child,0,new Set([root.id])))}</div>}</section>
  })}</div>
}

function ChatView({messages,busy,activeJobs,failedJobs,retryingJob,retry,send,addSource,courseSelected,openPlan}:{messages:ChatMessage[];busy:boolean;activeJobs:Job[];failedJobs:Job[];retryingJob:number|null;retry:(id:number)=>void;send:(text:string,files:File[],scope:"space_only"|"space_and_related"|"all_sources")=>void;addSource:()=>void;courseSelected:boolean;openPlan:()=>void}){
  const[draft,setDraft]=useState("")
  const[files,setFiles]=useState<File[]>([])
  const[fileError,setFileError]=useState("")
  const[retrievalScope,setRetrievalScope]=useState<"space_only"|"space_and_related"|"all_sources">("space_only")
  useEffect(()=>setRetrievalScope(courseSelected?"space_only":"all_sources"),[courseSelected])
  const inputRef=useRef<HTMLInputElement>(null)
  const endRef=useRef<HTMLDivElement>(null)
  useEffect(()=>{endRef.current?.scrollIntoView({behavior:"smooth",block:"end"})},[messages,busy])
  const submit=(event?:FormEvent)=>{event?.preventDefault();if(!draft.trim()&&!files.length)return;send(draft,files,retrievalScope);setDraft("");setFiles([]);setFileError("")}
  const onKeyDown=(event:KeyboardEvent<HTMLTextAreaElement>)=>{if(event.key==="Enter"&&!event.shiftKey){event.preventDefault();submit()}}
  const addFiles=(picked:FileList|null)=>{
    if(!picked)return
    const allowed=/\.(pdf|txt|md|docx|pptx|srt|vtt|json)$/i
    const rejected=Array.from(picked).filter(file=>!allowed.test(file.name))
    setFileError(rejected.length?"Choose a PDF, Word document or presentation, text, subtitle, or JSON file.":"")
    setFiles(previous=>[...previous,...Array.from(picked).filter(file=>allowed.test(file.name))])
  }
  return <section className="chat-layout" aria-label="Learning chat">
    <div className="conversation" aria-live="polite">
      {!messages.length?<div className="welcome">
        <div className="welcome-icon"><GraduationCap size={25}/></div>
        <p className="eyebrow">Your personal learning space</p>
        <h2>What would you like to learn?</h2>
        <p className="welcome-copy">Ask a question about your sources, or add a video or document. I’ll organize it for you.</p>
        <div className="suggestion-grid">
          <button onClick={()=>setDraft("Explain a difficult idea from my sources in simple terms.")}><span>Understand</span><small>Explain an idea simply</small></button>
          <button onClick={()=>setDraft("Summarize the main ideas in my learning library.")}><span>Review</span><small>Summarize my sources</small></button>
          <button onClick={openPlan}><span>Plan</span><small>Open your study plan</small></button>
          <button onClick={addSource}><span>Add knowledge</span><small>Choose a file or paste notes</small></button>
        </div>
      </div>:<div className="message-list">{messages.map(message=><article className={`message ${message.role}`} key={message.id}>
        <div className="message-avatar">{message.role==="assistant"?<Sparkles size={15}/>:<span>You</span>}</div>
        <div className="message-body"><div className="message-heading"><div className="message-author">{message.role==="assistant"?"LearnOS":"You"}</div>{message.role==="assistant"&&<CopyAnswer text={message.content}/>}</div><div className="message-text">{message.role==="assistant"?<MarkdownText text={message.content}/>:message.content}</div>
          {message.role==="assistant"&&message.evidenceCount!==undefined&&<details className="evidence-note"><summary>Sources used</summary><span>Grounded in {message.evidenceCount} passages from your library.{message.supplementalSourceCount?` Includes ${message.supplementalSourceCount} supplemental passages from outside this learning space (${message.supplementalSources?.join(", ")||"source names unavailable"}).`:""}</span></details>}
          {message.retryJobId&&<Button variant="outline" disabled={retryingJob===message.retryJobId} onClick={()=>retry(message.retryJobId!)}>{retryingJob===message.retryJobId?"Resuming…":"Resume processing"}</Button>}
        </div>
      </article>)}{busy&&<article className="message assistant"><div className="message-avatar"><Sparkles size={15}/></div><div className="message-body"><div className="message-author">LearnOS</div><div className="thinking"><LoaderCircle size={15}/> Working on it…</div></div></article>}<div ref={endRef}/></div>}
      {activeJobs.map(job=><div className="job-card" key={job.id}><div className="job-card-icon"><LoaderCircle size={16}/></div><div className="job-card-copy"><strong>Adding {sourceTitle(job)}</strong><span>{stageLabel(job.stage)} · {job.progress}</span></div><div className="job-meter"><i style={{width:`${progressValue(job.progress)}%`}}/></div></div>)}
      {failedJobs.map(job=><div className="job-card failed-card" key={job.id}><div className="job-card-icon"><X size={16}/></div><div className="job-card-copy"><strong>{sourceTitle(job)} needs attention</strong><span>{job.error_message||"Processing stopped before the source was ready."}</span></div><Button variant="outline" disabled={retryingJob===job.id} onClick={()=>retry(job.id)}>{retryingJob===job.id?"Resuming…":"Resume"}</Button></div>)}
    </div>
    <form className="composer-wrap" onSubmit={submit}>
      {files.length>0&&<div className="attachment-list">{files.map((file,index)=><span className="attachment" key={`${file.name}-${index}`}><FileText size={13}/>{file.name}<button type="button" aria-label={`Remove ${file.name}`} onClick={()=>setFiles(previous=>previous.filter((_,i)=>i!==index))}><X size={13}/></button></span>)}</div>}
      {fileError&&<div className="file-error" role="alert">{fileError}</div>}
      <div className="composer">
        <input ref={inputRef} hidden type="file" multiple accept=".pdf,.txt,.md,.docx,.pptx,.srt,.vtt,.json" onChange={event=>{addFiles(event.target.files);event.target.value=""}}/>
        <button className="composer-tool" type="button" title="Attach files" aria-label="Attach files" onClick={()=>inputRef.current?.click()}><Paperclip size={19}/></button>
        <textarea value={draft} rows={1} onChange={event=>setDraft(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask a question or paste a YouTube link…" aria-label="Ask LearnOS" />
        <button className="send-button" type="submit" disabled={busy||(!draft.trim()&&!files.length)} aria-label="Send"><Send size={17}/></button>
      </div>
      <div className="composer-hint"><label>Search scope <select className="text-input" value={retrievalScope} onChange={event=>setRetrievalScope(event.target.value as typeof retrievalScope)}>{courseSelected&&<><option value="space_only">This space only</option><option value="space_and_related">This space plus related sources</option></>}<option value="all_sources">All sources</option></select></label><span>Ask about your sources · attach a file · paste a YouTube link</span></div>
    </form>
  </section>
}

function progressValue(progress:string){const match=progress.match(/(\d+)\/(\d+)/);return match?Math.min(100,Math.round(Number(match[1])/Math.max(1,Number(match[2]))*100)):8}

function LibraryView({aiReady,courseName,sources,courses,jobs,failedJobs,retryingJob,retry,refresh,addSource,deleteSpace}:{aiReady:boolean;courseName?:string;sources:Source[];courses:Course[];jobs:Job[];failedJobs:Job[];retryingJob:number|null;retry:(id:number)=>void;refresh:()=>Promise<void>;addSource:()=>void;deleteSpace?:()=>void}){
  const[maintenanceNotice,setMaintenanceNotice]=useState("")
  const[maintenanceJobId,setMaintenanceJobId]=useState<number>()
  const[maintenanceProgress,setMaintenanceProgress]=useState<MaintenanceJob|null>(null)
  useEffect(()=>{let current=true;api.maintenanceJobs().then(rows=>{const latest=rows.find(job=>job.task==="organize_existing");if(current&&latest){setMaintenanceJobId(latest.id);setMaintenanceProgress(latest)}}).catch(()=>{});return()=>{current=false}},[])
  const organizeExisting=async()=>{try{const result=await api.organizeExisting(false);setMaintenanceJobId(result.id);setMaintenanceProgress({...result,task:"organize_existing",errors:{}});setMaintenanceNotice(result.progress_total?`Organization ${result.deduplicated?"is already active":"queued"}.`:"Everything is already organized.")}catch(e){setMaintenanceNotice(e instanceof Error?e.message:"Could not queue organization")}}
  useEffect(()=>{
    if(!maintenanceJobId||!maintenanceProgress||!["queued","running"].includes(maintenanceProgress.status))return
    let current=true
    const poll=async()=>{try{const rows=await api.maintenanceJobs();const latest=rows.find(job=>job.id===maintenanceJobId);if(current&&latest){setMaintenanceProgress(latest);if(latest.status==="done"){setMaintenanceNotice(latest.progress_total?`Organization finished: ${latest.progress_current}/${latest.progress_total} items processed.`:"Everything is already organized.");void refresh()}if(latest.status==="failed"){void refresh()}if(latest.status==="failed")setMaintenanceNotice(`Organization stopped with ${Object.keys(latest.errors||{}).length} error(s). Review the failed items below and retry.`)}}catch(e){if(current)setMaintenanceNotice(e instanceof Error?e.message:"Could not read organization progress")}}
    void poll();const timer=window.setInterval(()=>void poll(),1500);return()=>{current=false;window.clearInterval(timer)}
  },[maintenanceJobId,maintenanceProgress?.status])
  const retryOrganization=async()=>{if(!maintenanceJobId)return;try{await api.retryMaintenanceJob(maintenanceJobId);setMaintenanceProgress(value=>value?{...value,status:"queued",progress_current:0,errors:{}}:value);setMaintenanceNotice("Organization retry queued.")}catch(e){setMaintenanceNotice(e instanceof Error?e.message:"Could not retry organization")}}
  const latestBySource=new Map(jobs.map(job=>[job.source_id,job]))
  const failedIds=new Set(failedJobs.map(job=>job.source_id))
  const isFailed=(source:Source)=>source.status!=="needs_input"&&(source.status==="failed"||failedIds.has(source.id))
  const needsInput=sources.filter(source=>source.status==="needs_input")
  const failed=sources.filter(isFailed)
  const ready=sources.filter(source=>source.status==="done"&&!isFailed(source))
  const processing=sources.filter(source=>!needsInput.includes(source)&&!failed.includes(source)&&!ready.includes(source))
  const emptySpaces=courseName?[]:courses.filter(course=>course.source_count===0)
  const[clearing,setClearing]=useState(false)
  const removeEmptySpaces=async()=>{setClearing(true);try{for(const course of emptySpaces)await api.deleteCourse(course.id);await refresh()}catch(e){window.alert(e instanceof Error?e.message:"Could not remove the empty spaces")}finally{setClearing(false)}}
  const remove=async(source:Source,failedOnly=false)=>{const count=source.failed_attempt_count||0;const message=failedOnly?`Remove ${count} failed processing attempts for this video and keep the completed source?`:`Remove "${sourceTitle(source)}"? It is deleted from LearnOS together with everything made from it: its text, key ideas, practice questions and prepared study sessions. This cannot be undone.`;if(!window.confirm(message))return;try{await api.removeSource(source.id,failedOnly);await refresh()}catch(e){window.alert(e instanceof Error?e.message:"Could not remove source")}}
  const card=(source:Source)=><SourceCard key={source.id} aiReady={aiReady} source={source} job={latestBySource.get(source.id)} failed={isFailed(source)} courses={courses} retryingJob={retryingJob} retry={retry} remove={remove} refresh={refresh}/>
  const section=(title:string,help:string,items:Source[],attention=false)=>items.length>0&&<section className={attention?"attention-section":"library-section"}><div className="attention-heading"><div><h3>{title}</h3><p>{help}</p></div><Badge>{items.length}</Badge></div><div className="library-grid">{items.map(card)}</div></section>
  return <section className="library-view">
    <div className="library-intro"><div><p className="eyebrow">Your knowledge base</p><h2>Everything you've added</h2><p>Sources are grouped by what they need. Nothing is deleted automatically, so anything that failed can still be recovered.</p><p className="plan-help">LearnOS organizes your material in the background: it reads every source, files new ones into study spaces, and suggests a better place when the content says so. Nothing moves unless you choose.</p>{maintenanceNotice&&<p className="plan-help" role="status">{maintenanceNotice}</p>}{maintenanceProgress&&<div className="maintenance-progress" aria-live="polite"><div><strong>{maintenanceProgress.status==="done"?"Organization complete":maintenanceProgress.status==="failed"?"Organization needs attention":maintenanceProgress.status==="running"?"Organizing existing sources":"Organization queued"}</strong><span>{maintenanceProgress.progress_current} / {maintenanceProgress.progress_total} items{maintenanceProgress.checkpoint?` · last: ${maintenanceProgress.checkpoint}`:""}</span></div><progress max={Math.max(1,maintenanceProgress.progress_total)} value={maintenanceProgress.progress_current}/>{maintenanceProgress.status==="failed"&&<><ul>{Object.entries(maintenanceProgress.errors||{}).slice(0,3).map(([key,value])=><li key={key}>{key}: {value}</li>)}</ul><Button variant="outline" onClick={()=>void retryOrganization()}>Retry organization</Button></>}</div>}</div><div className="plan-actions">{deleteSpace&&<Button variant="ghost" onClick={deleteSpace}><Trash2 size={14}/>Delete this space</Button>}<Button variant="outline" disabled={maintenanceProgress!==null&&["queued","running"].includes(maintenanceProgress.status)} onClick={()=>void organizeExisting()}>{maintenanceProgress!==null&&["queued","running"].includes(maintenanceProgress.status)?<><LoaderCircle size={14}/>Organizing…</>:"Check everything now"}</Button><Button onClick={addSource}><Plus size={16}/>Add a source</Button></div></div>
    {emptySpaces.length>0&&<p className="plan-warning library-note" role="status">{emptySpaces.length===1?`The study space “${emptySpaces[0].name}” has no material.`:`${emptySpaces.length} study spaces have no material: ${emptySpaces.map(course=>course.name).join(", ")}.`} <button className="link-button" disabled={clearing} onClick={()=>void removeEmptySpaces()}>{emptySpaces.length===1?"Remove it":"Remove them"}</button></p>}
    {!sources.length?<div className="library-empty"><div className="welcome-icon"><FolderOpen size={23}/></div><h3>{courseName?`No material in ${courseName} yet`:"Your library is ready when you are"}</h3><p>{courseName?"Add material to this space, or choose All sources to see everything you have added.":"Add a video, document, or notes. LearnOS takes care of organizing it."}</p><Button onClick={addSource}><Plus size={16}/>{courseName?"Add material here":"Add your first material"}</Button></div>:<>
      {section("Needs your input","LearnOS could not read any text from these. Paste the text or a transcript, or remove them.",needsInput,true)}
      {section("Couldn't be processed","Processing stopped. Resume keeps the work already done.",failed,true)}
      {section("Processing","Being prepared in the background. You can keep working.",processing)}
      {section("Ready to study","Used by your study plans, study packs and chat.",ready)}
    </>}
  </section>
}

function SourceCard({aiReady,source,job,failed,courses,retryingJob,retry,remove,refresh}:{aiReady:boolean;source:Source;job?:Job;failed:boolean;courses:Course[];retryingJob:number|null;retry:(id:number)=>void;remove:(source:Source,failedOnly?:boolean)=>Promise<void>;refresh:()=>Promise<void>}){
  const[pasting,setPasting]=useState<string|null>(null)
  const[notice,setNotice]=useState("")
  const active=Boolean(job&&["running","retrying","queued"].includes(job.status))
  const issues=source.issues||[]
  const unreadable=issues.filter(issue=>issue.status==="needs_input").length
  const temporary=issues.filter(issue=>issue.status==="failed").length
  const status=failed?"failed":source.status==="needs_input"?"needs_input":active&&job?job.status:source.status
  const itemWord=(count:number)=>source.source_type==="youtube"?`video${count===1?"":"s"}`:`item${count===1?"":"s"}`
  const save=async(target:string,text:string,format:PasteFormat)=>{
    if(target==="source")await api.pasteSourceText(source.id,text,format)
    else await api.pasteDocumentTranscript(Number(target.slice(4)),text,format)
    setPasting(null);setNotice("Saved. LearnOS is indexing it now.");await refresh()
  }
  const reprocess=async()=>{try{await api.reprocessSource(source.id);setNotice("Trying again in the background.");await refresh()}catch(e){setNotice(e instanceof Error?e.message:"Could not try again")}}
  const[placing,setPlacing]=useState(false)
  const place=async(action:"move"|"show_in_both"|"keep",courseId:number)=>{setPlacing(true);setNotice("");try{await api.answerPlacement(source.id,action,courseId);await refresh()}catch(e){setNotice(e instanceof Error?e.message:"Could not change where this source is")}finally{setPlacing(false)}}
  const unshow=async(courseId:number)=>{setPlacing(true);setNotice("");try{await api.stopShowingSource(source.id,courseId);await refresh()}catch(e){setNotice(e instanceof Error?e.message:"Could not change where this source is shown")}finally{setPlacing(false)}}
  const suggestion=source.placement
  const target=suggestion?`“${suggestion.course_name||"another space"}”`:""
  return <article className={`source-card${failed||source.status==="needs_input"?" source-card-failed":""}`}>
    <div className="source-card-icon"><FileText size={18}/></div>
    <div className="source-card-content">
      <div className="source-card-title">{sourceTitle(source)}</div>
      <div className="source-card-subtitle">{source.description||source.topic_hint||source.domain_hint||source.name}</div>
      <div className="source-card-footer"><Badge>{statusLabel(status)}</Badge>{active&&job&&<span className="source-progress">{stageLabel(job.stage)} - {job.progress}</span>}</div>
      <div className="source-placement">{source.course_name?<>In <strong>{source.course_name}</strong></>:"Not in a study space yet"}{source.assignment_reason&&<span> · {source.assignment_reason}</span>}</div>
      {Boolean(source.also_in?.length)&&<div className="source-also-in">Also shown in {source.also_in!.map(space=><span className="space-chip" key={space.course_id}>{space.course_name||"a space"}<button type="button" disabled={placing} aria-label={`Stop showing ${source.name} in ${space.course_name||"this space"}`} onClick={()=>void unshow(space.course_id)}><X size={12}/></button></span>)}</div>}
      {suggestion&&<div className="placement-suggestion" role="status"><strong>{suggestion.kind==="move"?`Looks like it belongs in ${target}`:suggestion.kind==="also"?`Also fits ${target}`:`Fits ${target}`}</strong><span>{suggestion.reason}{suggestion.leaves_space_empty?" Moving it would leave its current space empty.":""}</span><div className="placement-actions">
        {suggestion.kind==="add"?<><Button variant="outline" disabled={placing} onClick={()=>void place("move",suggestion.course_id)}>Add to {target}</Button><Button variant="ghost" disabled={placing} onClick={()=>void place("keep",suggestion.course_id)}>Not there</Button></>
        :<><Button variant="outline" disabled={placing} onClick={()=>void place(suggestion.kind==="move"?"move":"show_in_both",suggestion.course_id)}>{suggestion.kind==="move"?`Move to ${target}`:"Show in both"}</Button><Button variant="ghost" disabled={placing} onClick={()=>void place(suggestion.kind==="move"?"show_in_both":"move",suggestion.course_id)}>{suggestion.kind==="move"?"Show in both":`Move to ${target}`}</Button><Button variant="ghost" disabled={placing} onClick={()=>void place("keep",suggestion.course_id)}>Keep here</Button></>}
      </div></div>}
      {source.status==="done"&&!active&&(source.concepts_pending?<div className="source-coverage pending">{aiReady?"Not fully read for key ideas yet. LearnOS reads it in the background.":"Key ideas need an AI model: add GEMINI_API_KEY to the .env file and restart LearnOS; it then reads them by itself."}</div>:source.sections_pending?<div className="source-coverage pending">Sections not built yet. LearnOS finishes this in the background.</div>:Boolean(source.section_count)&&<div className="source-coverage">{source.section_count} section{source.section_count===1?"":"s"} · {source.concept_count||0} key idea{source.concept_count===1?"":"s"}</div>)}
      {failed&&<div className="source-error">{job?.error_message||"Processing stopped before this source was ready."}</div>}
      {source.status==="needs_input"&&source.status_message&&<div className="source-error">{source.status_message}</div>}
      {source.status==="done"&&issues.length>0&&<div className="source-issue-note">{[unreadable?`${unreadable} ${itemWord(unreadable)} without captions`:"",temporary?`${temporary} not fetched yet (temporary)`:""].filter(Boolean).join(" · ")}{source.document_count?` out of ${source.document_count}`:""}. The rest is ready to study.</div>}
      {issues.length>0&&<details className="source-issues" open={source.status==="needs_input"||undefined}><summary>{issues.length===1?`Show the ${itemWord(1)} that needs help`:`Show the ${issues.length} ${itemWord(issues.length)} that need help`}</summary><ul>{issues.map(issue=><li key={issue.document_id}><div><strong>{issue.title}</strong>{issue.message&&<span>{issue.message}</span>}</div>{pasting===`doc:${issue.document_id}`?<PasteTextForm label={`Transcript for “${issue.title}”`} onCancel={()=>setPasting(null)} onSave={(text,format)=>save(`doc:${issue.document_id}`,text,format)}/>:<Button variant="outline" disabled={active} onClick={()=>setPasting(`doc:${issue.document_id}`)}>Paste transcript</Button>}</li>)}</ul></details>}
      {pasting==="source"&&<PasteTextForm label="Text or transcript for this source" onCancel={()=>setPasting(null)} onSave={(text,format)=>save("source",text,format)}/>}
      {notice&&<p className="plan-help" role="status">{notice}</p>}
      {Boolean(source.failed_attempt_count)&&<div className="duplicate-note">{source.failed_attempt_count} failed attempt{source.failed_attempt_count===1?"":"s"} for this video</div>}
    </div>
    <select aria-label={`Move ${source.name} to a learning space`} className="source-course-select" value={source.course_id||""} onChange={async event=>{try{await api.moveSource(source.id,event.target.value?Number(event.target.value):null);await refresh()}catch(e){window.alert(e instanceof Error?e.message:"Could not move source")}}}><option value="">All sources (unassigned)</option>{courses.map(course=><option key={course.id} value={course.id}>{course.name}</option>)}</select>
    <div className="source-card-actions">
      {failed&&job&&<Button variant="outline" disabled={retryingJob===job.id} onClick={()=>retry(job.id)}>{retryingJob===job.id?<><LoaderCircle size={14}/>Resuming...</>:<><RefreshCw size={14}/>Resume</>}</Button>}
      {source.status==="needs_input"&&source.source_type!=="youtube"&&pasting!=="source"&&<Button variant="outline" onClick={()=>setPasting("source")}>Paste text</Button>}
      {issues.length>0&&!active&&<Button variant="outline" onClick={()=>void reprocess()}><RefreshCw size={14}/>{source.source_type==="youtube"?"Retry fetching":"Try again"}</Button>}
      {Boolean(source.failed_attempt_count)&&<Button variant="outline" onClick={()=>void remove(source,true)}><Trash2 size={14}/>Remove {source.failed_attempt_count} failed</Button>}
      <Button variant="outline" onClick={()=>void remove(source)}><Trash2 size={14}/>Remove</Button>
    </div>
  </article>
}

function PasteTextForm({label,onSave,onCancel}:{label:string;onSave:(text:string,format:PasteFormat)=>Promise<void>;onCancel:()=>void}){
  const[text,setText]=useState("")
  const[format,setFormat]=useState<PasteFormat>("plain")
  const[busy,setBusy]=useState(false)
  const[error,setError]=useState("")
  const submit=async(event:FormEvent)=>{event.preventDefault();if(!text.trim())return;setBusy(true);setError("");try{await onSave(text,format)}catch(e){setError(e instanceof Error?e.message:"Could not save the text")}finally{setBusy(false)}}
  return <form className="paste-form" onSubmit={event=>void submit(event)}>
    <label className="field-label">{label}<textarea className="text-input paste-input" value={text} onChange={event=>setText(event.target.value)} placeholder="Paste the text here…" autoFocus/></label>
    <div className="paste-form-actions"><select className="text-input" aria-label="Text format" value={format} onChange={event=>setFormat(event.target.value as PasteFormat)}><option value="plain">Plain text</option><option value="srt">SRT subtitles</option><option value="vtt">VTT subtitles</option></select><Button type="submit" disabled={busy||!text.trim()}>{busy?<><LoaderCircle size={14}/>Saving…</>:"Save and index"}</Button><Button variant="ghost" onClick={onCancel}>Cancel</Button></div>
    {error&&<p className="dialog-error" role="alert">{error}</p>}
  </form>
}

function AboutYou({saved}:{saved:()=>void}){
  const[profile,setProfile]=useState<Profile>({background:"",goal:"",practice:"questions"})
  // "failed": what was saved could not be read, so saving is held back rather than replacing it with blanks.
  const[state,setState]=useState<"loading"|"failed"|"idle"|"saving"|"saved">("loading")
  const[error,setError]=useState("")
  const[attempt,setAttempt]=useState(0)
  useEffect(()=>{let current=true;setState("loading");setError("");api.profile().then(result=>{if(current){setProfile(result);setState("idle")}}).catch(()=>{if(current){setState("failed");setError("What you told your tutor could not be loaded.")}});return()=>{current=false}},[attempt])
  const edit=(change:Partial<Profile>)=>{setProfile(previous=>({...previous,...change}));if(state==="saved")setState("idle")}
  const save=async(event:FormEvent)=>{event.preventDefault();setState("saving");setError("");try{setProfile(await api.saveProfile(profile));setState("saved");saved()}catch(e){setError(e instanceof Error?e.message:"Could not save");setState("idle")}}
  return <form className="about-form" onSubmit={event=>void save(event)}>
    <label>Your background<textarea className="text-input" rows={2} maxLength={600} value={profile.background} placeholder="For example: MSc in data science, comfortable with Python and machine learning" onChange={event=>edit({background:event.target.value})}/></label>
    <label>Your goal<input className="text-input" maxLength={300} value={profile.goal} placeholder="For example: build and evaluate AI agents" onChange={event=>edit({goal:event.target.value})}/></label>
    <label>How you like to practise<select className="text-input" value={profile.practice} onChange={event=>edit({practice:event.target.value as Profile["practice"]})}><option value="questions">Check questions</option><option value="coding">Small coding tasks</option><option value="both">Both</option></select></label>
    <div className="plan-actions"><Button type="submit" disabled={state==="loading"||state==="saving"||state==="failed"}>{state==="saving"?"Saving…":"Save"}</Button>{state==="saved"&&<span className="plan-help" role="status">Saved. It is added to every study session you copy.</span>}{error&&<span className="plan-error" role="alert">{error}{state==="failed"&&<> <button type="button" className="link-button" onClick={()=>setAttempt(value=>value+1)}>Try again</button></>}</span>}</div>
  </form>
}

function DeleteSpaceDialog({course,close,done}:{course:Course;close:()=>void;done:()=>void}){
  const[busy,setBusy]=useState(false)
  const[error,setError]=useState("")
  const count=course.source_count||0
  const run=async(withMaterial:boolean)=>{setBusy(true);setError("");try{await api.deleteCourse(course.id,withMaterial);done()}catch(e){setError(e instanceof Error?e.message:"Could not delete this space")}finally{setBusy(false)}}
  return <div className="dialog-backdrop" onMouseDown={event=>event.currentTarget===event.target&&close()}><section className="add-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-space-title">
    <header className="dialog-header"><div><h2 id="delete-space-title">Delete “{course.name}”?</h2><p>{count?`It has ${plural(count,"source")}.`:"It has no material."}</p></div><button className="icon-button" aria-label="Close" onClick={close}><X size={18}/></button></header>
    <div className="dialog-content"><p className="plan-help">{count?"Keep its material and it stays in your library, under All sources. Delete the material too and it is removed from this computer, with everything LearnOS made from it.":"Its study plan and progress are removed."}</p>{error&&<p className="dialog-error" role="alert">{error}</p>}</div>
    <footer className="dialog-footer"><Button variant="ghost" onClick={close}>Cancel</Button>{count?<><Button variant="outline" disabled={busy} onClick={()=>void run(false)}>Keep the material</Button><Button variant="danger" disabled={busy} onClick={()=>void run(true)}>Delete the material too</Button></>:<Button variant="danger" disabled={busy} onClick={()=>void run(false)}>Delete the space</Button>}</footer>
  </section></div>
}

function AddSourceDialog({courseId,courses,close,refresh}:{courseId?:number;courses:Course[];close:()=>void;refresh:()=>Promise<void>}){
  const[mode,setMode]=useState<AddMode>("file")
  const[files,setFiles]=useState<File[]>([])
  const[url,setUrl]=useState("")
  const[name,setName]=useState("")
  const[pasted,setPasted]=useState("")
  const[busy,setBusy]=useState(false)
  const[notice,setNotice]=useState("")
  const[error,setError]=useState("")
  const[spaceMode,setSpaceMode]=useState<"auto"|"existing"|"new"|"unassigned">(courseId?"existing":"auto")
  const[spaceId,setSpaceId]=useState(courseId?String(courseId):"")
  const[spaceName,setSpaceName]=useState("")
  const fileRef=useRef<HTMLInputElement>(null)
  const submit=async(event:FormEvent)=>{
    event.preventDefault();setBusy(true);setError("");setNotice("")
    try{
      if(mode==="file"&&!files.length)throw new Error("Choose a file first.")
      if(mode==="youtube"&&!url.trim())throw new Error("Paste a YouTube video or playlist link.")
      if(mode==="paste"&&!pasted.trim())throw new Error("Paste some notes or a transcript first.")
      let targetCourseId:number|undefined
      if(spaceMode==="existing"){
        targetCourseId=Number(spaceId)
        if(!targetCourseId)throw new Error("Choose a learning space.")
      }else if(spaceMode==="new"){
        if(!spaceName.trim())throw new Error("Enter a name for the new learning space.")
        targetCourseId=(await api.createCourse({name:spaceName.trim()})).id
      }
      if(mode==="file"){
        if(!files.length)throw new Error("Choose a file first.")
        const assignmentMode=spaceMode==="auto"?"auto":spaceMode==="unassigned"?"explicit_unassigned":"explicit_course"
        const results=await api.upload(files,{courseId:targetCourseId,assignmentMode});const conflict=results.some((result:any)=>result.assignment_conflict);setNotice(conflict?"At least one source already exists in a different space. Existing assignments were preserved; move it from Library if you want to change it.":`Added ${files.length===1?"1 file":`${files.length} files`}. I’m preparing them now.`);setFiles([])
      }else if(mode==="youtube"){
        if(!url.trim())throw new Error("Paste a YouTube video or playlist link.")
        const result=await api.youtube(url,{courseId:targetCourseId,assignmentMode:spaceMode==="auto"?"auto":spaceMode==="unassigned"?"explicit_unassigned":"explicit_course",name:name||undefined})
        setNotice(result.deduplicated?result.assignment_conflict?"This source is already filed in a different space. Its existing assignment was preserved; move it from Library if you want to change it.":result.status==="done"?"That source is already in your library and ready.":"That source is already in your library.":"Added. I’ll fetch and prepare it in the background.")
      }else{
        if(!pasted.trim())throw new Error("Paste some notes or a transcript first.")
        const form=new FormData();form.append("name",name||"Pasted notes");form.append("payload",pasted);form.append("input_format","plain");form.append("assignment_mode",spaceMode==="auto"?"auto":spaceMode==="unassigned"?"explicit_unassigned":"explicit_course");if(targetCourseId)form.append("course_id",String(targetCourseId))
        const result:any=await api.manual(form);setNotice(result.deduplicated?result.assignment_conflict?"These notes already exist in a different space. Their existing assignment was preserved.":"These notes are already in your library.":"Added. I’ll organize it in the background.");setPasted("")
      }
      await refresh()
    }catch(e){setError(e instanceof Error?e.message:"Could not add this source")}finally{setBusy(false)}
  }
  return <div className="dialog-backdrop" onMouseDown={event=>event.currentTarget===event.target&&close()}><section className="add-dialog" role="dialog" aria-modal="true" aria-labelledby="add-title">
    <header className="dialog-header"><div><h2 id="add-title">Add to your library</h2><p>LearnOS will prepare it for you.</p></div><button className="icon-button" aria-label="Close" onClick={close}><X size={18}/></button></header>
    <div className="dialog-tabs" role="tablist">{(["file","youtube","paste"] as AddMode[]).map(item=><button key={item} role="tab" aria-selected={mode===item} className={mode===item?"active":""} onClick={()=>setMode(item)}>{item==="file"?"File":item==="youtube"?"YouTube":"Paste notes"}</button>)}</div>
    <form onSubmit={submit}>
      <div className="dialog-content">
        {mode==="file"&&<><input ref={fileRef} type="file" hidden multiple accept=".pdf,.txt,.md,.docx,.pptx,.srt,.vtt,.json" onChange={event=>setFiles(Array.from(event.target.files||[]))}/><button className="file-drop" type="button" onClick={()=>fileRef.current?.click()}><UploadCloud size={23}/><strong>{files.length?`${files.length} file${files.length===1?"":"s"} selected`:"Choose files"}</strong><span>PDF, Word, PowerPoint, text, subtitle, or JSON</span></button>{files.map(file=><div className="file-row" key={file.name}><FileText size={15}/>{file.name}</div>)}</>}
        {mode==="youtube"&&<label className="field-label">Paste a YouTube link<input className="text-input" autoFocus value={url} onChange={event=>setUrl(event.target.value)} placeholder="https://youtu.be/…"/></label>}
        {mode==="paste"&&<label className="field-label">Paste your notes or transcript<textarea className="text-input paste-input" autoFocus value={pasted} onChange={event=>setPasted(event.target.value)} placeholder="Paste text here…"/></label>}
        {(mode!=="file")&&<label className="field-label optional-label">Name <span>optional</span><input className="text-input" value={name} onChange={event=>setName(event.target.value)} placeholder="e.g. Week 2 lecture"/></label>}
        <label className="field-label">Study space<select className="text-input" value={spaceMode} onChange={event=>setSpaceMode(event.target.value as "auto"|"existing"|"new"|"unassigned")}><option value="auto">Let LearnOS decide (recommended)</option>{courses.length>0&&<option value="existing">Put it in a space I choose</option>}<option value="new">Create a new space</option><option value="unassigned">Keep it in All sources</option></select></label>
        {spaceMode==="existing"&&<label className="field-label">Choose a space<select className="text-input" value={spaceId} onChange={event=>setSpaceId(event.target.value)}><option value="">Select a space</option>{courses.map(course=><option key={course.id} value={course.id}>{course.name}</option>)}</select></label>}
        {spaceMode==="new"&&<label className="field-label">New space name<input className="text-input" value={spaceName} onChange={event=>setSpaceName(event.target.value)} placeholder="e.g. Natural Language Processing"/></label>}
        {notice&&<p className="dialog-notice">{notice}</p>}{error&&<p className="dialog-error" role="alert">{error}</p>}
      </div>
      <footer className="dialog-footer"><Button variant="ghost" onClick={close}>Close</Button><Button type="submit" disabled={busy}>{busy?<><LoaderCircle size={15}/>Adding…</>:<><Plus size={15}/>Add source</>}</Button></footer>
    </form>
  </section></div>
}
