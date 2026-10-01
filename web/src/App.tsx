import {useEffect,useMemo,useRef,useState,type ReactNode} from "react"
import {BookOpen,BrainCircuit,FileText,FolderOpen,GraduationCap,Library,LoaderCircle,Plus,RefreshCw,Sparkles,UploadCloud,X} from "lucide-react"
import {api,type Course,type Job,type Source,type Stats} from "./lib/api"

type View="home"|"library"|"study"
type Mode="files"|"paste"|"youtube"
const nav=[["home","Home",BrainCircuit],["library","Library",Library],["study","Study",GraduationCap]] as const

function Button({children,onClick,variant="primary",disabled=false}:{children:ReactNode;onClick?:()=>void;variant?:"primary"|"outline"|"ghost";disabled?:boolean}){
  return <button className={"btn "+variant} onClick={onClick} disabled={disabled}>{children}</button>
}
function Badge({children}:{children:ReactNode}){return <span className="badge">{children}</span>}
function progressValue(s:string){const m=s.match(/(\d+)\/(\d+)/);return m?Math.min(100,Math.round(Number(m[1])/Math.max(1,Number(m[2]))*100)):6}

export default function App(){
  const[view,setView]=useState<View>("home")
  const[courses,setCourses]=useState<Course[]>([])
  const[courseId,setCourseId]=useState<number>()
  const[sources,setSources]=useState<Source[]>([])
  const[jobs,setJobs]=useState<Job[]>([])
  const[stats,setStats]=useState<Stats|null>(null)
  const[health,setHealth]=useState<any>()
  const[add,setAdd]=useState(false)
  const[error,setError]=useState("")
  const refresh=async()=>{
    try{
      setError("")
      const[c,j,h]=await Promise.all([api.courses(),api.jobs(),api.health()])
      setCourses(c);setJobs(j);setHealth(h)
      if(courseId){
        const[s,st]=await Promise.all([api.sources(courseId),api.stats(courseId)])
        setSources(s);setStats(st)
      }else{
        setSources(await api.sources())
        setStats(null)
      }
    }catch(e){setError(e instanceof Error?e.message:"Backend unavailable")}
  }
  useEffect(()=>{refresh()},[courseId])
  useEffect(()=>{
    if(!jobs.some(j=>j.status==="queued"||j.status==="running"))return
    const t=setInterval(refresh,3500)
    return()=>clearInterval(t)
  },[jobs,courseId])
  const course=useMemo(()=>courses.find(c=>c.id===courseId),[courses,courseId])
  const active=jobs.filter(j=>j.status==="queued"||j.status==="running")
  return <div className="shell">
    <aside className="rail">
      <div className="brand"><div className="mark"><Sparkles size={15}/></div><span>LearnOS</span></div>
      <div className="nav">{nav.map(([id,label,Icon])=><button key={id} className={view===id?"active":""} onClick={()=>setView(id)}><Icon size={17}/><span>{label}</span></button>)}</div>
      <div className="rail-foot"><Button onClick={()=>setAdd(true)}><Plus size={16}/><span>Add</span></Button><small className="muted" style={{padding:"0 9px"}}>{health?.llm?"AI connected":"Local fallback"}</small></div>
    </aside>
    <main className="main">
      <header className="top">
        <div><div className="eyebrow">{course?.domain||"Personal learning"}</div><strong>{course?.name||"All sources"}</strong></div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          {active.length>0&&<Badge><LoaderCircle size={11}/>{active.length} running</Badge>}
          <button className="btn outline" onClick={refresh}><RefreshCw size={14}/></button>
        </div>
      </header>
      <div className="workspace">
        {error&&<div className="error" style={{marginBottom:14}}>{error}</div>}
        {view==="home"&&<Home stats={stats} sources={sources} jobs={jobs} add={()=>setAdd(true)} study={()=>setView("study")}/>}
        {view==="library"&&<LibraryView sources={sources}/>}
        {view==="study"&&<StudyView courseId={courseId}/>}
      </div>
    </main>
    {add&&<AddMaterial courseId={courseId} close={()=>setAdd(false)} refresh={refresh}/>}
  </div>
}

function Home({stats,sources,jobs,add,study}:{stats:Stats|null;sources:Source[];jobs:Job[];add:()=>void;study:()=>void}){
  const current=jobs.find(j=>j.status==="running")||jobs.find(j=>j.status==="queued")
  return <>
    <div className="hero">
      <section className="card hero-card">
        <div className="eyebrow">Your learning workspace</div>
        <h1>Add knowledge as you find it. LearnOS organizes it over time.</h1>
        <p className="muted" style={{maxWidth:650,lineHeight:1.7}}>Add a PDF today, paste a lecture transcript tomorrow, or add a playlist later. Nothing has to arrive together.</p>
        <div className="actions"><Button onClick={add}><UploadCloud size={16}/>Add material</Button><Button onClick={study} variant="outline"><BookOpen size={16}/>Study</Button></div>
      </section>
      <section className="card panel">
        <div className="eyebrow">Background agent</div><h2 style={{marginTop:7}}>Processing</h2>
        {current?<><div style={{marginTop:20,fontWeight:650,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{current.source_name}</div><div className="muted" style={{fontSize:11,margin:"4px 0 9px"}}>{current.stage} · {current.progress}</div><div className="progress"><i style={{width:progressValue(current.progress)+"%"}}/></div><p className="muted" style={{fontSize:11,lineHeight:1.55}}>Keep adding material while this runs. Progress is stored.</p></>:<div className="empty">Nothing is processing right now.</div>}
      </section>
    </div>
    <div className="stats">
      {[["Sources",stats?.sources??sources.length],["Documents",stats?.documents??"-"],["Chunks",stats?.chunks??"-"],["Groups",stats?.grouped??"-"]].map(([a,b])=><div className="card stat" key={String(a)}><strong>{String(b)}</strong><span>{String(a)}</span></div>)}
    </div>
  </>
}

function LibraryView({sources}:{sources:Source[]}){
  return <>
    <div className="section-head"><div><h2>Library</h2><div className="muted" style={{fontSize:12}}>Add one thing now and another later.</div></div></div>
    <div className="list">
      {sources.map(s=><div className="card row" key={s.id}>
        <div style={{width:36,height:36,borderRadius:9,background:"#f0f0f5",display:"grid",placeItems:"center"}}><FileText size={16}/></div>
        <div className="row-main"><b>{s.name}</b><p>{s.description||s.topic_hint||"No extra context."}</p><div className="chips">{s.domain_hint&&<Badge>{s.domain_hint}</Badge>}<Badge>{s.source_type}</Badge><Badge>{s.status}</Badge></div></div>
      </div>)}
      {!sources.length&&<div className="card empty"><FolderOpen size={28}/><div style={{marginTop:8}}>Your library is empty.</div></div>}
    </div>
  </>
}

function StudyView({courseId}:{courseId?:number}){
  const[topic,setTopic]=useState(""),[question,setQuestion]=useState(""),[syllabus,setSyllabus]=useState(""),[pack,setPack]=useState(""),[answer,setAnswer]=useState(""),[busy,setBusy]=useState(false),[error,setError]=useState("")
  const run=async(fn:()=>Promise<void>)=>{setBusy(true);setError("");try{await fn()}catch(e){setError(e instanceof Error?e.message:"Request failed")}finally{setBusy(false)}}
  return <>
    <div className="section-head"><div><h2>Study</h2><div className="muted" style={{fontSize:12}}>Evidence comes from your indexed learning sources.</div></div></div>
    {error&&<div className="error" style={{marginBottom:12}}>{error}</div>}
    <div className="panel-grid">
      <section className="card panel">
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}><div><strong>Learning path</strong><div className="muted" style={{fontSize:11}}>Syllabus from your course</div></div><Button variant="outline" disabled={busy} onClick={()=>run(async()=>setSyllabus((await api.syllabus(courseId)).syllabus))}><Sparkles size={14}/>Generate</Button></div>
        <div className="study-output" style={{marginTop:12,background:"#f7f7fa",borderRadius:11,padding:13,minHeight:300,overflow:"auto"}}>{syllabus||"Generate a syllabus after your sources are indexed."}</div>
      </section>
      <section className="card panel">
        <strong>Study one topic</strong><div className="muted" style={{fontSize:11}}>Retrieve, rerank, expand context, synthesize.</div>
        <div className="actions"><input className="input" value={topic} onChange={e=>setTopic(e.target.value)} placeholder="e.g. hybrid retrieval"/><Button disabled={busy||!topic} onClick={()=>run(async()=>setPack((await api.studyPack(topic,courseId)).study_text))}>Build</Button></div>
        <textarea className="textarea study-output" style={{marginTop:10,minHeight:220}} readOnly value={pack} placeholder="Your study pack appears here."/>
        <div style={{marginTop:15}}><strong>Ask your library</strong><textarea className="textarea" style={{marginTop:7,minHeight:86}} value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a focused question..."/><Button disabled={busy||!question} onClick={()=>run(async()=>setAnswer((await api.study(question,courseId)).answer))}><BookOpen size={14}/>Ask</Button></div>
        {answer&&<div className="study-output" style={{marginTop:10,background:"#f7f7fa",borderRadius:11,padding:13}}>{answer}</div>}
      </section>
    </div>
  </>
}

function AddMaterial({courseId,close,refresh}:{courseId?:number;close:()=>void;refresh:()=>Promise<void>}){
  const[mode,setMode]=useState<Mode>("files"),[files,setFiles]=useState<File[]>([]),[drag,setDrag]=useState(false),[name,setName]=useState(""),[payload,setPayload]=useState(""),[format,setFormat]=useState("plain"),[url,setUrl]=useState(""),[topic,setTopic]=useState(""),[domain,setDomain]=useState(""),[description,setDescription]=useState(""),[busy,setBusy]=useState(false),[message,setMessage]=useState(""),[error,setError]=useState("")
  const inputRef=useRef<HTMLInputElement>(null)
  const pick=(fs:File[])=>setFiles(prev=>[...prev,...fs])
  const submit=async()=>{
    setBusy(true);setError("");setMessage("")
    try{
      if(mode==="files"){
        if(!files.length)throw new Error("Choose at least one file.")
        await api.upload(files,{courseId,topic,domain,description})
        setFiles([]);setMessage("Added. The agent will process it in the background.")
      }else if(mode==="paste"){
        if(!payload.trim())throw new Error("Paste some content first.")
        const f=new FormData();f.append("name",name||"Pasted material");f.append("payload",payload);f.append("input_format",format)
        if(courseId)f.append("course_id",String(courseId));if(topic)f.append("topic_hint",topic);if(domain)f.append("domain_hint",domain);if(description)f.append("description",description)
        await api.manual(f);setPayload("");setMessage("Added. You can add another item whenever you like.")
      }else{
        if(!url.trim())throw new Error("Enter a YouTube URL.")
        await api.youtube(url,{courseId,name:name||undefined,topic,domain,description});setUrl("");setMessage("Added. The video or playlist is now queued.")
      }
      await refresh()
    }catch(e){setError(e instanceof Error?e.message:"Could not add material")}finally{setBusy(false)}
  }
  return <div className="modal-bg" onMouseDown={e=>e.currentTarget===e.target&&close()}>
    <div className="modal">
      <div className="modal-head"><div><strong>Add material</strong><div className="muted" style={{fontSize:11,marginTop:2}}>Add one thing now. Nothing has to arrive together.</div></div><button className="btn ghost" onClick={close}><X size={17}/></button></div>
      <div className="modal-body">
        <div className="tabs">{([["files","Files"],["paste","Paste"],["youtube","YouTube"]] as const).map(([id,label])=><button key={id} className={mode===id?"active":""} onClick={()=>setMode(id)}>{label}</button>)}</div>
        {mode==="files"&&<div className={"drop "+(drag?"drag":"")} onClick={()=>inputRef.current?.click()} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);pick(Array.from(e.dataTransfer.files))}}><UploadCloud size={24}/><strong>Drop files here</strong><small>or click to choose · one file or many</small><input ref={inputRef} hidden type="file" multiple accept=".pdf,.txt,.md,.docx,.srt,.vtt,.json" onChange={e=>pick(Array.from(e.target.files||[]))}/>{files.length>0&&<div className="chips" style={{justifyContent:"center"}}>{files.map((f,i)=><Badge key={i}>{f.name}</Badge>)}</div>}</div>}
        {mode==="paste"&&<><div className="field"><label>Name <span className="muted">(optional)</span></label><input className="input" value={name} onChange={e=>setName(e.target.value)} placeholder="Lecture 3 — Retrieval"/></div><div className="field"><label>Format</label><select className="select" value={format} onChange={e=>setFormat(e.target.value)}><option value="plain">Plain transcript / text</option><option value="srt">SRT</option><option value="vtt">VTT</option><option value="python_dict">Python dictionary</option><option value="json">JSON</option></select></div><div className="field"><label>Content</label><textarea className="textarea" style={{minHeight:220}} value={payload} onChange={e=>setPayload(e.target.value)} placeholder={format==="python_dict"?"Paste a Python dict containing transcript/text/content...":"Paste transcript, notes, or source text here."}/></div></>}
        {mode==="youtube"&&<div className="field"><label>YouTube video or playlist URL</label><input className="input" value={url} onChange={e=>setUrl(e.target.value)} placeholder="https://youtube.com/..."/></div>}
        <details style={{marginTop:14}}><summary className="muted" style={{cursor:"pointer",fontSize:12}}>Optional context</summary><div className="two"><div className="field"><label>Topic</label><input className="input" value={topic} onChange={e=>setTopic(e.target.value)} placeholder="RAG"/></div><div className="field"><label>Domain</label><input className="input" value={domain} onChange={e=>setDomain(e.target.value)} placeholder="AI / Information Retrieval"/></div></div><div className="field"><label>Description</label><textarea className="textarea" style={{minHeight:70}} value={description} onChange={e=>setDescription(e.target.value)} placeholder="Optional. Helps routing."/></div></details>
        {message&&<div className="notice" style={{marginTop:13}}>{message}</div>}{error&&<div className="error" style={{marginTop:13}}>{error}</div>}
      </div>
      <div className="modal-foot"><Button variant="ghost" onClick={close}>Close</Button><Button disabled={busy} onClick={submit}>{busy?<><LoaderCircle size={14}/>Adding…</>:<><Plus size={14}/>Add to library</>}</Button></div>
    </div>
  </div>
}