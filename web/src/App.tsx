import {useEffect,useMemo,useRef,useState,type FormEvent,type KeyboardEvent,type ReactNode} from "react"
import {Brain,Check,Copy,FileText,FolderOpen,GraduationCap,Library,LoaderCircle,MessageSquare,Paperclip,Plus,RefreshCw,Send,Sparkles,Trash2,UploadCloud,X} from "lucide-react"
import {api,type ChatTurn,type ConversationSummary,type Course,type Job,type MaintenanceJob,type PasteFormat,type PlanStatus,type Source,type StudyPackExport} from "./lib/api"
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

function Button({children,onClick,variant="primary",disabled=false,type="button"}:{children:ReactNode;onClick?:()=>void;variant?:"primary"|"outline"|"ghost";disabled?:boolean;type?:"button"|"submit"}){
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

function CopyAnswer({text}:{text:string}){
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
  return <button type="button" className="copy-answer" onClick={()=>void copy()} aria-label={copied?"Answer copied":"Copy answer"} title={copied?"Copied":"Copy answer"}>{copied?<Check size={14}/>:<Copy size={14}/>}<span>{copied?"Copied":"Copy"}</span></button>
}

export default function App(){
  const[view,setView]=useState<View>("plan")
  const[courses,setCourses]=useState<Course[]>([])
  const[courseId,setCourseId]=useState<number>()
  const[sources,setSources]=useState<Source[]>([])
  const[jobs,setJobs]=useState<Job[]>([])
  const[messages,setMessages]=useState<ChatMessage[]>([])
  const[conversations,setConversations]=useState<ConversationSummary[]>([])
  const[conversationId,setConversationId]=useState<string>()
  const conversationIdRef=useRef<string|undefined>(undefined)
  const[busy,setBusy]=useState(false)
  const[retryingJob,setRetryingJob]=useState<number|null>(null)
  const[addOpen,setAddOpen]=useState(false)
  const[error,setError]=useState("")
  const[health,setHealth]=useState<any>()
  const[planRefreshKey,setPlanRefreshKey]=useState(0)
  const[planRegenerateKey,setPlanRegenerateKey]=useState(0)
  const previousJobStates=useRef(new Map<number,string>())
  const refreshSequence=useRef(0)
  const chatOpenSequence=useRef(0)

  const refresh=async()=>{
    const requestId=++refreshSequence.current
    try{
      setError("")
      const[c,j,h,s]=await Promise.all([api.courses(),api.jobs(),api.health(),api.sources(courseId)])
      if(requestId!==refreshSequence.current)return
      const terminal=new Set(["done","failed"])
      const becameReady=j.some(job=>previousJobStates.current.get(job.id)!==undefined&&
        !terminal.has(previousJobStates.current.get(job.id)!)&&terminal.has(job.status))
      previousJobStates.current=new Map(j.map(job=>[job.id,job.status]))
      setCourses(c);setJobs(j);setHealth(h);setSources(s)
      if(becameReady)setPlanRefreshKey(key=>key+1)
    }catch(e){setError(e instanceof Error?e.message:"Could not connect to LearnOS")}
  }
  useEffect(()=>{refreshSequence.current++;void refresh()},[courseId])
  useEffect(()=>{let current=true;api.conversations().then(async loaded=>{if(!current)return;let rows=loaded;if(!rows.length&&!localStorage.getItem("learnos.chat-migration-v1")){try{const old=JSON.parse(localStorage.getItem("learnos.current-chat")||"[]") as ChatMessage[];if(old.length){const created=await api.createConversation();for(const message of old)await api.addConversationMessage(created.id,{role:message.role,content:message.content},{evidenceCount:message.evidenceCount,supplementalSourceCount:message.supplementalSourceCount,supplementalSources:message.supplementalSources,retryJobId:message.retryJobId});localStorage.setItem("learnos.chat-migration-v1","done");rows=await api.conversations()}}catch{setError("Your earlier chat is still stored in this browser; LearnOS could not migrate it yet.")}}if(!current)return;setConversations(rows);const saved=localStorage.getItem("learnos.active-conversation");const selected=rows.find(row=>row.id===saved)||rows[0];if(selected){conversationIdRef.current=selected.id;setConversationId(selected.id);const full=await api.getConversation(selected.id);if(current)setMessages(full.messages.map(m=>({id:String(m.id),role:m.role,content:m.content,evidenceCount:m.metadata?.evidenceCount,supplementalSourceCount:m.metadata?.supplementalSourceCount,supplementalSources:m.metadata?.supplementalSources,retryJobId:m.metadata?.retryJobId})))}}).catch(e=>setError(e instanceof Error?e.message:"Could not load conversations"));return()=>{current=false}},[])
  useEffect(()=>{if(conversationId)localStorage.setItem("learnos.active-conversation",conversationId)},[conversationId])
  const currentJobs=useMemo(()=>latestJobs(jobs),[jobs])
  const activeJobs=currentJobs.filter(job=>["queued","running","retrying"].includes(job.status))
  const failedJobs=currentJobs.filter(job=>job.status==="failed")
  useEffect(()=>{
    if(!activeJobs.length)return
    const timer=window.setInterval(()=>void refresh(),3000)
    return()=>window.clearInterval(timer)
  },[activeJobs.length,courseId])

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
      <button className="brand" onClick={()=>setView("chat")}><span className="brand-mark"><Sparkles size={17}/></span><span>LearnOS</span></button>
      <Button variant="outline" disabled={busy} onClick={()=>void newChat()}><Plus size={16}/>New chat</Button>
      <div className="sidebar-section"><div className="sidebar-label">Recent chats</div>{conversations.slice(0,12).map(chat=><div className="source-shortcut" key={chat.id}><button className="text-link" disabled={busy} onClick={()=>void openChat(chat.id)} title={chat.title}>{chat.title}</button><button className="icon-button" disabled={busy} aria-label="Delete conversation" onClick={()=>void removeChat(chat.id)}><Trash2 size={13}/></button></div>)}{!conversations.length&&<p className="sidebar-empty">Your conversations will appear here.</p>}</div>
      <nav className="primary-nav" aria-label="Main navigation">
        <button className={view==="chat"?"selected":""} onClick={()=>setView("chat")}><MessageSquare size={17}/>Chat</button>
        <button className={view==="plan"?"selected":""} onClick={()=>setView("plan")}><GraduationCap size={17}/>Study plan</button>
        <button className={view==="learn"?"selected":""} onClick={()=>setView("learn")}><Brain size={17}/>Practice</button>
        <button className={view==="library"?"selected":""} onClick={()=>setView("library")}><Library size={17}/>Library <span className="nav-count">{sources.length}</span></button>
      </nav>
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
      {view==="chat"?<ChatView messages={messages} busy={busy} activeJobs={activeJobs} failedJobs={failedJobs} retryingJob={retryingJob} retry={retryJob} send={sendMessage} addSource={()=>setAddOpen(true)} courseSelected={courseId!==undefined}/>:view==="learn"?<LearnView courseId={courseId}/>:view==="plan"?<StudyPlanView courseId={courseId} refreshKey={planRefreshKey} regenerateKey={planRegenerateKey} addSource={()=>setAddOpen(true)}/>:<LibraryView aiReady={health?.llm===true} sources={sources} courses={courses} jobs={currentJobs} failedJobs={failedJobs} retryingJob={retryingJob} retry={retryJob} refresh={refresh} addSource={()=>setAddOpen(true)}/>}
    </main>
    {addOpen&&<AddSourceDialog courseId={courseId} courses={courses} close={()=>setAddOpen(false)} refresh={refresh}/>}
  </div>
}

function StudyPlanView({courseId,refreshKey,regenerateKey,addSource}:{courseId?:number;refreshKey:number;regenerateKey:number;addSource:()=>void}){
  const handledRegenerate=useRef(regenerateKey)
  const[syllabus,setSyllabus]=useState("")
  const[planWarnings,setPlanWarnings]=useState<string[]>([])
  const[relationshipInfo,setRelationshipInfo]=useState<PlanRelationships>({sources:[],cross_paths:[],shared_foundations:[]})
  const[topic,setTopic]=useState("")
  const[topicId,setTopicId]=useState<string>()
  const[planNodes,setPlanNodes]=useState<OutlineTopic[]>([])
  const[tutor,setTutor]=useState("general")
  const[maxInputTokens,setMaxInputTokens]=useState(6000)
  const[pack,setPack]=useState<StudyPackExport|null>(null)
  const[expandedOutline,setExpandedOutline]=useState<string|null>(null)
  const[completedTopics,setCompletedTopics]=useState<Set<string>>(new Set())
  const[busy,setBusy]=useState(false)
  const[error,setError]=useState("")
  const[planStatus,setPlanStatus]=useState<PlanStatus>("ready")
  const[planError,setPlanError]=useState("")
  const[generatedAt,setGeneratedAt]=useState<string|null>(null)
  const[completedIds,setCompletedIds]=useState<Set<string>>(new Set())
  const[pollTick,setPollTick]=useState(0)
  const activeCourse=useRef(courseId)
  const applyPlan=(plan:Awaited<ReturnType<typeof api.syllabus>>,progress:Awaited<ReturnType<typeof api.studyProgress>>)=>{
    setSyllabus(plan.syllabus||"");setPlanWarnings(plan.warnings||[]);setRelationshipInfo(normalizePlanRelationships(plan.source_relationships))
    setPlanStatus(plan.status||"ready");setPlanError(plan.error||"");setGeneratedAt(plan.generated_at||null)
    setPlanNodes((plan.nodes||[]).map(node=>({id:node.id,parentId:node.parent_id||"",pathId:node.path_id||"main",kind:node.type,topic:node.title,purpose:node.metadata?.purpose||"",prerequisites:node.metadata?.prerequisites||"",groups:Array.isArray(node.metadata?.source_groups)?node.metadata.source_groups:String(node.metadata?.source_groups||"").split(",").map(s=>s.trim()).filter(Boolean),why:node.metadata?.why_included||"",enables:node.metadata?.enables||"",sourceIds:Array.isArray(node.metadata?.source_ids)?node.metadata.source_ids:[],sourceRoles:Array.isArray(node.metadata?.source_roles)?node.metadata.source_roles:[],confidence:node.metadata?.topic_merge_confidence||0})))
    setCompletedTopics(new Set((progress.completed_legacy_topics??progress.completed_topics).map(normalizedTopic)));setCompletedIds(new Set(progress.completed_topic_ids||[]))
  }
  useEffect(()=>{let current=true;if(activeCourse.current!==courseId){setPlanNodes([]);setPlanStatus("ready");setPlanError("");setGeneratedAt(null);setCompletedIds(new Set());setCompletedTopics(new Set());setExpandedOutline(null)}activeCourse.current=courseId;setSyllabus("");setPlanWarnings([]);setRelationshipInfo({sources:[],cross_paths:[],shared_foundations:[]});setPack(null);setTopicId(undefined);setTopic("");setBusy(true);setError("");const regenerate=regenerateKey!==handledRegenerate.current;handledRegenerate.current=regenerateKey;Promise.all([api.syllabus(courseId,regenerate),api.studyProgress(courseId)]).then(([plan,progress])=>{if(current)applyPlan(plan,progress)}).catch(e=>{if(current)setError(e instanceof Error?e.message:"Could not load your study plan")}).finally(()=>{if(current)setBusy(false)});return()=>{current=false}},[courseId,refreshKey,regenerateKey])
  // While a plan is being built in the background, check again every few seconds; reading never builds.
  useEffect(()=>{if(planStatus!=="updating")return;const requested=courseId;const timer=window.setTimeout(()=>{Promise.all([api.syllabus(requested),api.studyProgress(requested)]).then(([plan,progress])=>{if(activeCourse.current===requested){applyPlan(plan,progress);setPollTick(tick=>tick+1)}}).catch(()=>{if(activeCourse.current===requested)setPollTick(tick=>tick+1)})},4000);return()=>window.clearTimeout(timer)},[planStatus,pollTick,courseId])
  const makePack=async(event:FormEvent)=>{event.preventDefault();if(!topic.trim())return;setBusy(true);setError("");try{const result=await api.exportStudyPack(topic.trim(),courseId,tutor,maxInputTokens,topicId);setPack(result)}catch(e){setError(e instanceof Error?e.message:"Could not prepare study pack")}finally{setBusy(false)}}
  const outline=planNodes
  const topicOptions=outline.filter(item=>item.kind==="study").map(item=>item.topic)
  const isDone=(item:OutlineTopic)=>completedIds.has(item.id)||completedTopics.has(normalizedTopic(item.topic))
  const toggleComplete=async(item:OutlineTopic,completed:boolean)=>{const key=normalizedTopic(item.topic);const next=new Set(completedTopics);const nextIds=new Set(completedIds);if(completed){next.add(key);nextIds.add(item.id)}else{next.delete(key);nextIds.delete(item.id)}setCompletedTopics(next);setCompletedIds(nextIds);try{await api.updateStudyProgress(item.topic,courseId,completed,item.id)}catch(e){setCompletedTopics(completedTopics);setCompletedIds(completedIds);setError(e instanceof Error?e.message:"Could not save your progress")}}
  const selectTopic=(item:OutlineTopic)=>{if(item.kind==="outline"){setExpandedOutline(expandedOutline===item.id?null:item.id);setPack(null);return}setTopic(item.topic);setTopicId(item.id);setPack(null);window.setTimeout(()=>document.getElementById("study-pack-form")?.scrollIntoView({behavior:"smooth",block:"center"}),0)}
  const hasPlan=planStatus!=="empty"&&(planNodes.length>0||Boolean(syllabus&&!syllabus.startsWith("No ")))
  const builtAt=generatedAt?new Date(generatedAt.endsWith("Z")||generatedAt.includes("+")?generatedAt:generatedAt+"Z").toLocaleString():""
  const statusLine=planStatus==="updating"?"Updating in the background with your latest sources. The last version is shown until it is ready.":planStatus==="failed"?planError:planStatus==="outdated"?"Your sources changed since this plan was built. An update will start shortly.":builtAt?`Up to date · built ${builtAt}`:"Up to date"
  const topicsByGroup=new Map<string,{title:string;items:OutlineTopic[]}>()
  outline.forEach(item=>{const current=topicsByGroup.get(item.pathId)||{title:item.pathId,items:[]};if(!item.parentId)current.title=item.topic;current.items.push(item);topicsByGroup.set(item.pathId,current)})
  return <section className="plan-view">
    <div className="plan-intro"><p className="eyebrow">Learn from your own sources</p><h2>Start with a course. Follow its outline.</h2><p>Add one or more sources and put them in a course. LearnOS turns them into a topic outline. Pick a topic to get a short, cited text pack that you can study here or paste into an AI tutor.</p></div>
    {!hasPlan&&planStatus==="updating"?<div className="plan-empty"><h3><LoaderCircle size={15} className="learn-spin"/> Building your study plan in the background…</h3><p>LearnOS is turning your sources into an outline. The first plan for a space can take a few minutes; later updates only process what changed. You can keep working, and this page updates when it is ready.</p></div>:!hasPlan&&planStatus==="failed"?<div className="plan-empty"><h3>The study plan could not be built</h3><p className="plan-error" role="alert">{planError}</p></div>:!hasPlan?<div className="plan-empty"><h3>{busy?"Loading your study plan…":"Start here: add your learning material"}</h3><p>Choose or create a study space, add sources, and LearnOS will organize them and prepare the outline automatically. You can leave this page while processing continues.</p><div className="plan-actions"><Button onClick={addSource}><Plus size={16}/>Add sources</Button></div>{error&&<p className="plan-error" role="alert">{error}</p>}{syllabus&&<p className="plan-error">{syllabus}</p>}</div>:<>
      <div className="plan-section"><div className="plan-section-heading"><div><span className="step-number">1</span><div><h3>Course outline</h3><span className="plan-help">Broad topics open their own outline. Choose a leaf topic to prepare its study pack.</span></div></div><div className="plan-actions"><CopyAnswer text={syllabus}/></div></div>
        <p className={`plan-status ${planStatus}`} role="status">{planStatus==="updating"&&<LoaderCircle size={13} className="learn-spin"/>}{statusLine}</p>
        {planWarnings.length>0&&<div className="plan-warning" role="status"><strong>This plan is incomplete.</strong><ul>{planWarnings.map((warning,index)=><li key={index}>{warning}</li>)}</ul><span>This is usually a temporary problem with the AI service. Press Refresh (top right) to try again.</span></div>}
        {(relationshipInfo.sources.length>0||relationshipInfo.shared_foundations.length>0)&&<details className="outline-markdown"><summary>How LearnOS grouped these sources</summary><div className="plan-help">Source identity is inferred and may be uncertain. Shared topics alone do not establish that sources belong to one course.</div>{relationshipInfo.sources.map((r:any,index:number)=><p key={`${r.source_a_id}-${r.source_b_id}-${index}`}>Source {r.source_a_id} + source {r.source_b_id}: <strong>{String(r.relation_type||"").replaceAll("_"," ")}</strong> ({Math.round(Number(r.confidence||0)*100)}% confidence). {r.rationale}</p>)}{relationshipInfo.shared_foundations.length>0&&<p>Shared foundations: {relationshipInfo.shared_foundations.join(", ")}</p>}</details>}
        <div className="outline-tree">{[...topicsByGroup.entries()].map(([group,path])=>{const done=path.items.filter(item=>item.kind==="study"&&isDone(item)).length;return <details className="outline-group" key={group} open><summary><span>{path.title}</span><small>{done} topics complete</small></summary><div className="outline-topic-list">{path.items.filter(item=>!item.parentId||item.parentId.toLowerCase()==="none").map(item=>{const done=isDone(item);return <article className={`outline-topic${done?" completed":""}`} key={`${group}-${item.id}`}>
          {item.kind==="study"&&<label className="outline-check"><input type="checkbox" checked={done} onChange={event=>void toggleComplete(item,event.target.checked)} aria-label={`Mark ${item.topic} ${done?"not complete":"complete"}`}/><span>{done?<Check size={13}/>:null}</span></label>}
          <button className="outline-topic-button" onClick={()=>selectTopic(item)}><strong>{item.topic}</strong><small>{item.kind==="outline"?"Learning path or topic outline · open subtopics":item.purpose}</small>{item.why&&<em>Why included: {item.why}</em>}{item.enables&&<em>Enables: {item.enables}</em>}{item.sourceRoles.length>0&&<em>Source roles: {item.sourceRoles.join(", ")}</em>}{item.prerequisites&&item.prerequisites!=="None recorded"&&item.prerequisites.toLowerCase()!=="none"&&<em>Prerequisites: {item.prerequisites} · confidence {Math.round(item.confidence*100)}%</em>}</button>
        </article>})}</div></details>})}</div>
        {expandedOutline&&<div className="plan-suboutline"><div className="plan-section-heading"><div><h3>{outline.find(item=>item.id===expandedOutline)?.topic}</h3><span className="plan-help">Study these focused topics one at a time.</span></div><Button variant="ghost" onClick={()=>setExpandedOutline(null)}>Close outline</Button></div><div className="outline-topic-list">{outline.filter(item=>item.parentId===expandedOutline).map(item=><article className={`outline-topic${isDone(item)?" completed":""}`} key={item.id}>{item.kind==="study"&&<label className="outline-check"><input type="checkbox" checked={isDone(item)} onChange={event=>void toggleComplete(item,event.target.checked)} aria-label={`Mark ${item.topic} ${isDone(item)?"not complete":"complete"}`}/><span>{isDone(item)?<Check size={13}/>:null}</span></label>}<button className="outline-topic-button" onClick={()=>selectTopic(item)}><strong>{item.topic}</strong><small>{item.kind==="outline"?"Open sub-outline":item.purpose}</small></button></article>)}</div></div>}
        <details className="outline-markdown"><summary>View / copy the full syllabus text</summary><div className="plan-syllabus"><MarkdownText text={syllabus}/></div></details>
      </div>
      <div className="plan-section" id="study-pack-form"><div className="plan-section-heading"><div><span className="step-number">2</span><h3>Prepare one topic to study</h3></div></div><p className="plan-help">LearnOS finds source passages for this topic. The pack has a configurable source size; it is an estimate because ChatGPT, Gemini, and Claude tokenize text differently.</p><form className="pack-form" onSubmit={event=>void makePack(event)}><label className="field-label">Topic<input className="text-input" list="syllabus-topics" value={topic} onChange={event=>setTopic(event.target.value)} placeholder="Choose an outline topic or type one" required/><datalist id="syllabus-topics">{topicOptions.map((value,index)=><option key={`${value}-${index}`} value={value}/>)}</datalist></label><label className="field-label">AI tutor<select className="text-input" value={tutor} onChange={event=>setTutor(event.target.value)}><option value="general">Any AI tutor</option><option value="chatgpt">ChatGPT</option><option value="gemini">Gemini</option><option value="claude">Claude</option></select></label><label className="field-label">Source pack size<select className="text-input" value={maxInputTokens} onChange={event=>setMaxInputTokens(Number(event.target.value))}><option value={4000}>Compact · about 4k tokens</option><option value={6000}>Balanced · about 6k tokens</option><option value={10000}>Expanded · about 10k tokens</option></select></label><Button type="submit" disabled={busy||!topic.trim()}>{busy?<><LoaderCircle size={15}/>Preparing…</>:<><Sparkles size={15}/>Prepare study pack</>}</Button></form>
        {error&&<p className="plan-error" role="alert">{error}</p>}{pack&&<div className="study-pack-result"><div className="plan-section-heading"><h3>Ready to copy</h3><CopyAnswer text={pack.study_text}/></div><p className="plan-help">About {pack.estimated_tokens.toLocaleString()} tokens · {pack.sources.length} passages{pack.omitted_passages?` · ${pack.omitted_passages} more passages did not fit`:""}. Copy the instructions and cited excerpts together.</p>{pack.omitted_passages>0&&<div className="pack-subtopics"><strong>This topic is broader than one pack.</strong><p>Narrow it to one of these related topics, or select a larger pack size.</p>{pack.suggested_subtopics.map((value,index)=><button type="button" key={`${value}-${index}`} onClick={()=>{setTopic(value);setTopicId(undefined);setPack(null)}}>{value}</button>)}</div>}<div className="plan-syllabus"><MarkdownText text={pack.study_text}/></div></div>}
      </div>
    </>}
  </section>
}

function ChatView({messages,busy,activeJobs,failedJobs,retryingJob,retry,send,addSource,courseSelected}:{messages:ChatMessage[];busy:boolean;activeJobs:Job[];failedJobs:Job[];retryingJob:number|null;retry:(id:number)=>void;send:(text:string,files:File[],scope:"space_only"|"space_and_related"|"all_sources")=>void;addSource:()=>void;courseSelected:boolean}){
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
          <button onClick={()=>setDraft("Make a study plan based on what I have learned so far.")}><span>Plan</span><small>Build a study plan</small></button>
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

function LibraryView({aiReady,sources,courses,jobs,failedJobs,retryingJob,retry,refresh,addSource}:{aiReady:boolean;sources:Source[];courses:Course[];jobs:Job[];failedJobs:Job[];retryingJob:number|null;retry:(id:number)=>void;refresh:()=>Promise<void>;addSource:()=>void}){
  const[maintenanceNotice,setMaintenanceNotice]=useState("")
  const[maintenanceJobId,setMaintenanceJobId]=useState<number>()
  const[maintenanceProgress,setMaintenanceProgress]=useState<MaintenanceJob|null>(null)
  useEffect(()=>{let current=true;api.maintenanceJobs().then(rows=>{const latest=rows.find(job=>job.task==="organize_existing");if(current&&latest){setMaintenanceJobId(latest.id);setMaintenanceProgress(latest)}}).catch(()=>{});return()=>{current=false}},[])
  const organizeExisting=async()=>{try{const result=await api.organizeExisting(true);setMaintenanceJobId(result.id);setMaintenanceProgress({...result,task:"organize_existing",errors:{}});setMaintenanceNotice(result.progress_total?`Organization ${result.deduplicated?"is already active":"queued"}.`:"Everything is already organized.")}catch(e){setMaintenanceNotice(e instanceof Error?e.message:"Could not queue organization")}}
  useEffect(()=>{
    if(!maintenanceJobId||!maintenanceProgress||!["queued","running"].includes(maintenanceProgress.status))return
    let current=true
    const poll=async()=>{try{const rows=await api.maintenanceJobs();const latest=rows.find(job=>job.id===maintenanceJobId);if(current&&latest){setMaintenanceProgress(latest);if(latest.status==="done"){setMaintenanceNotice(latest.progress_total?`Organization finished: ${latest.progress_current}/${latest.progress_total} items processed.`:"Everything is already organized.");void refresh()}if(latest.status==="failed")setMaintenanceNotice(`Organization stopped with ${Object.keys(latest.errors||{}).length} error(s). Review the failed items below and retry.`)}}catch(e){if(current)setMaintenanceNotice(e instanceof Error?e.message:"Could not read organization progress")}}
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
  const remove=async(source:Source,failedOnly=false)=>{const count=source.failed_attempt_count||0;const message=failedOnly?`Remove ${count} failed processing attempts for this video and keep the completed source?`:`Remove "${sourceTitle(source)}" and all duplicate records with its indexed study data?`;if(!window.confirm(message))return;try{await api.removeSource(source.id,failedOnly);await refresh()}catch(e){window.alert(e instanceof Error?e.message:"Could not remove source")}}
  const card=(source:Source)=><SourceCard key={source.id} aiReady={aiReady} source={source} job={latestBySource.get(source.id)} failed={isFailed(source)} courses={courses} retryingJob={retryingJob} retry={retry} remove={remove} refresh={refresh}/>
  const section=(title:string,help:string,items:Source[],attention=false)=>items.length>0&&<section className={attention?"attention-section":"library-section"}><div className="attention-heading"><div><h3>{title}</h3><p>{help}</p></div><Badge>{items.length}</Badge></div><div className="library-grid">{items.map(card)}</div></section>
  return <section className="library-view">
    <div className="library-intro"><div><p className="eyebrow">Your knowledge base</p><h2>Everything you've added</h2><p>Sources are grouped by what they need. Nothing is deleted automatically, so anything that failed can still be recovered.</p><p className="plan-help">Each source shows the study space it is in and why. Use the space menu on a source to move it. “Organize existing” reads older sources in full for their key ideas and files unassigned sources; your own choices are kept.</p>{maintenanceNotice&&<p className="plan-help" role="status">{maintenanceNotice}</p>}{maintenanceProgress&&<div className="maintenance-progress" aria-live="polite"><div><strong>{maintenanceProgress.status==="done"?"Organization complete":maintenanceProgress.status==="failed"?"Organization needs attention":maintenanceProgress.status==="running"?"Organizing existing sources":"Organization queued"}</strong><span>{maintenanceProgress.progress_current} / {maintenanceProgress.progress_total} items{maintenanceProgress.checkpoint?` · last: ${maintenanceProgress.checkpoint}`:""}</span></div><progress max={Math.max(1,maintenanceProgress.progress_total)} value={maintenanceProgress.progress_current}/>{maintenanceProgress.status==="failed"&&<><ul>{Object.entries(maintenanceProgress.errors||{}).slice(0,3).map(([key,value])=><li key={key}>{key}: {value}</li>)}</ul><Button variant="outline" onClick={()=>void retryOrganization()}>Retry organization</Button></>}</div>}</div><div className="plan-actions"><Button variant="outline" disabled={maintenanceProgress!==null&&["queued","running"].includes(maintenanceProgress.status)} onClick={()=>void organizeExisting()}>{maintenanceProgress!==null&&["queued","running"].includes(maintenanceProgress.status)?<><LoaderCircle size={14}/>Organizing…</>:"Organize existing"}</Button><Button onClick={addSource}><Plus size={16}/>Add a source</Button></div></div>
    {!sources.length?<div className="library-empty"><div className="welcome-icon"><FolderOpen size={23}/></div><h3>Your library is ready when you are</h3><p>Add a video, document, or transcript. LearnOS will take care of organizing it.</p><Button onClick={addSource}><Plus size={16}/>Add your first source</Button></div>:<>
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
  return <article className={`source-card${failed||source.status==="needs_input"?" source-card-failed":""}`}>
    <div className="source-card-icon"><FileText size={18}/></div>
    <div className="source-card-content">
      <div className="source-card-title">{sourceTitle(source)}</div>
      <div className="source-card-subtitle">{source.description||source.topic_hint||source.domain_hint||source.name}</div>
      <div className="source-card-footer"><Badge>{statusLabel(status)}</Badge>{active&&job&&<span className="source-progress">{stageLabel(job.stage)} - {job.progress}</span>}</div>
      <div className="source-placement">{source.course_name?<>In <strong>{source.course_name}</strong></>:"Not in a study space yet"}{source.assignment_reason&&<span> · {source.assignment_reason}</span>}</div>
      {source.status==="done"&&!active&&(source.concepts_pending?<div className="source-coverage pending">{aiReady?"Not fully read for key ideas yet. Use “Organize existing” to read it in full.":"Key ideas need an AI model: add GEMINI_API_KEY, then use “Organize existing”."}</div>:source.sections_pending?<div className="source-coverage pending">Sections not built yet. Use “Organize existing” to finish indexing.</div>:Boolean(source.section_count)&&<div className="source-coverage">{source.section_count} section{source.section_count===1?"":"s"} · {source.concept_count||0} key idea{source.concept_count===1?"":"s"}</div>)}
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
        <label className="field-label">Learning space<select className="text-input" value={spaceMode} onChange={event=>setSpaceMode(event.target.value as "auto"|"existing"|"new"|"unassigned")}><option value="auto">Let LearnOS decide after processing</option><option value="existing">Add to an existing space</option><option value="new">Create a new space</option><option value="unassigned">Keep in All Sources</option></select></label>
        {spaceMode==="existing"&&<label className="field-label">Choose a space<select className="text-input" value={spaceId} onChange={event=>setSpaceId(event.target.value)}><option value="">Select a learning space</option>{courses.map(course=><option key={course.id} value={course.id}>{course.name}</option>)}</select></label>}
        {spaceMode==="new"&&<label className="field-label">New space name<input className="text-input" value={spaceName} onChange={event=>setSpaceName(event.target.value)} placeholder="e.g. Natural Language Processing"/></label>}
        {mode==="file"&&<><input ref={fileRef} type="file" hidden multiple accept=".pdf,.txt,.md,.docx,.pptx,.srt,.vtt,.json" onChange={event=>setFiles(Array.from(event.target.files||[]))}/><button className="file-drop" type="button" onClick={()=>fileRef.current?.click()}><UploadCloud size={23}/><strong>{files.length?`${files.length} file${files.length===1?"":"s"} selected`:"Choose files"}</strong><span>PDF, Word, PowerPoint, text, subtitle, or JSON</span></button>{files.map(file=><div className="file-row" key={file.name}><FileText size={15}/>{file.name}</div>)}</>}
        {mode==="youtube"&&<label className="field-label">Paste a YouTube link<input className="text-input" autoFocus value={url} onChange={event=>setUrl(event.target.value)} placeholder="https://youtu.be/…"/></label>}
        {mode==="paste"&&<label className="field-label">Paste your notes or transcript<textarea className="text-input paste-input" autoFocus value={pasted} onChange={event=>setPasted(event.target.value)} placeholder="Paste text here…"/></label>}
        {(mode!=="file")&&<label className="field-label optional-label">Name <span>optional</span><input className="text-input" value={name} onChange={event=>setName(event.target.value)} placeholder="e.g. Week 2 lecture"/></label>}
        {notice&&<p className="dialog-notice">{notice}</p>}{error&&<p className="dialog-error" role="alert">{error}</p>}
      </div>
      <footer className="dialog-footer"><Button variant="ghost" onClick={close}>Close</Button><Button type="submit" disabled={busy}>{busy?<><LoaderCircle size={15}/>Adding…</>:<><Plus size={15}/>Add source</>}</Button></footer>
    </form>
  </section></div>
}
