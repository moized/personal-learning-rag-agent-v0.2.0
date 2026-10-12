import {useCallback,useEffect,useState,type FormEvent} from "react"
import {Check,LoaderCircle,RotateCcw,Sparkles,Target} from "lucide-react"
import {api,type ConceptMastery,type DueReview,type Misconception,type NextTopic,type QuizQuestion,type QuizResult} from "./lib/api"

type ActiveQuiz={conceptId:number;concept:string;question:QuizQuestion;result?:QuizResult}
type Overview={mastery:ConceptMastery[];due:DueReview[];next:NextTopic[];misconceptions:Misconception[]}

const STATUS_LABEL:Record<ConceptMastery["status"],string>={unseen:"Not started",learning:"Learning",due:"Review due",mastered:"Mastered"}
// Recall ratings follow SM-2's 0-5 scale; the labels are the usual flashcard buttons.
const RATINGS=[{label:"Forgot",value:1},{label:"Hard",value:3},{label:"Good",value:4},{label:"Easy",value:5}]

function percent(value:number){return `${Math.round(value*100)}%`}
function dueLabel(iso:string|null){
  if(!iso)return "Not scheduled"
  const days=(new Date(iso+"Z").getTime()-Date.now())/86400000
  if(days<=0)return "Due now"
  if(days<1)return "Due today"
  return `Due in ${Math.round(days)} day${Math.round(days)===1?"":"s"}`
}

export default function LearnView({courseId,sourceCount=0,readyCount=0,aiReady=true,addSource,openPlan,openLibrary}:{courseId?:number;sourceCount?:number;readyCount?:number;aiReady?:boolean;addSource?:()=>void;openPlan?:()=>void;openLibrary?:()=>void}){
  const[data,setData]=useState<Overview>({mastery:[],due:[],next:[],misconceptions:[]})
  const[loading,setLoading]=useState(true)
  const[error,setError]=useState("")
  const[quiz,setQuiz]=useState<ActiveQuiz|null>(null)
  const[answer,setAnswer]=useState("")
  const[working,setWorking]=useState(false)
  const[starting,setStarting]=useState<number|null>(null)

  const load=useCallback(async()=>{
    try{
      const[mastery,due,next,misconceptions]=await Promise.all([api.learnerMastery(courseId),api.learnerDue(courseId),api.learnerNext(courseId),api.misconceptions()])
      setData({mastery:mastery.concepts,due:due.due,next:next.next,misconceptions:misconceptions.misconceptions})
      setError("")
    }catch(e){setError(e instanceof Error?e.message:"Could not load your progress")}
    finally{setLoading(false)}
  },[courseId])
  useEffect(()=>{setLoading(true);setQuiz(null);setAnswer("");void load()},[load])

  const startQuiz=async(conceptId:number,concept:string)=>{
    setStarting(conceptId);setError("")
    try{
      const generated=await api.generateQuiz(conceptId,courseId)
      setQuiz({conceptId,concept,question:generated.items[0]});setAnswer("")
      window.setTimeout(()=>document.getElementById("learn-quiz")?.scrollIntoView({behavior:"smooth",block:"center"}),0)
    }catch(e){setError(e instanceof Error?e.message:"Could not create a question")}
    finally{setStarting(null)}
  }
  const submitAnswer=async(event:FormEvent)=>{
    event.preventDefault()
    if(!quiz||!answer.trim()||working)return
    setWorking(true);setError("")
    try{
      const result=await api.answerQuiz(quiz.question.id,answer.trim())
      setQuiz({...quiz,result});await load()
    }catch(e){setError(e instanceof Error?e.message:"Could not grade your answer")}
    finally{setWorking(false)}
  }
  const rate=async(item:DueReview,rating:number)=>{
    try{await api.selfRating(item.concept_id,rating,courseId);await load()}
    catch(e){setError(e instanceof Error?e.message:"Could not save your rating")}
  }
  const resolve=async(id:number)=>{
    try{await api.resolveMisconception(id);await load()}
    catch(e){setError(e instanceof Error?e.message:"Could not update this item")}
  }

  const counts=data.mastery.reduce((acc,item)=>{acc[item.status]++;return acc},{unseen:0,learning:0,due:0,mastered:0} as Record<ConceptMastery["status"],number>)
  const empty=!loading&&!data.mastery.length

  return <section className="plan-view learn-view">
    <div className="plan-intro"><p className="eyebrow">Learn for the long term</p><h2>Practice what you've studied.</h2><p>LearnOS quizzes you on the ideas in your sources, tracks how well you know each one, and schedules reviews just before you would forget.</p></div>
    {error&&<p className="plan-error" role="alert">{error}</p>}
    {loading?<div className="plan-empty"><h3><LoaderCircle size={15} className="learn-spin"/> Loading your progress…</h3></div>
    :empty?<div className="plan-empty"><h3>Nothing to practise yet</h3><p>{!sourceCount?"Add your learning material first. Practice quizzes you on the key ideas LearnOS finds in it.":!readyCount?"None of your material could be read yet. Open the Library to see what it needs.":!aiReady?"Practice uses the key ideas in your material, and finding them needs an AI model: add GEMINI_API_KEY to the .env file and restart LearnOS.":"LearnOS is still reading your material for its key ideas. They appear here for quizzes and review as soon as they are found."}</p><div className="plan-actions">{!sourceCount&&addSource?<button type="button" className="btn primary" onClick={addSource}>Add material</button>:!readyCount&&openLibrary?<button type="button" className="btn outline" onClick={openLibrary}>Open the Library</button>:openPlan?<button type="button" className="btn outline" onClick={openPlan}>Go to your study plan</button>:null}</div></div>
    :<>
      <div className="learn-summary" aria-label="Progress summary">
        {(["mastered","learning","due","unseen"] as const).map(status=><div className={`learn-stat ${status}`} key={status}><strong>{counts[status]}</strong><span>{STATUS_LABEL[status]}</span></div>)}
      </div>

      {quiz&&<div className="plan-section learn-quiz" id="learn-quiz">
        <div className="plan-section-heading"><div><span className="step-number"><Target size={13}/></span><div><h3>{quiz.concept}</h3><span className="plan-help">Answer in your own words.</span></div></div><button className="text-link" onClick={()=>setQuiz(null)}>Close</button></div>
        <p className="learn-question">{quiz.question.question}</p>
        {!quiz.result?<form onSubmit={event=>void submitAnswer(event)}>
          <textarea className="text-input paste-input" value={answer} onChange={event=>setAnswer(event.target.value)} placeholder="Type your answer…" aria-label="Your answer" autoFocus/>
          <div className="plan-actions learn-actions"><button className="btn primary" type="submit" disabled={working||!answer.trim()}>{working?<><LoaderCircle size={15} className="learn-spin"/>Checking…</>:<><Check size={15}/>Check answer</>}</button></div>
        </form>:<div className={`learn-result ${quiz.result.correct?"correct":"incorrect"}`} role="status">
          <div className="learn-result-head"><strong>{quiz.result.correct?"Good work":"Not quite yet"}</strong><span className="badge">Score {percent(quiz.result.score)}</span><span className="badge">Mastery {percent(quiz.result.mastery)}</span><span className="badge">{dueLabel(quiz.result.next_due_at)}</span></div>
          {quiz.result.feedback&&<p>{quiz.result.feedback}</p>}
          <details open={!quiz.result.correct}><summary>Reference answer</summary><p>{quiz.result.reference_answer}</p>{quiz.result.evidence_chunk_ids.length>0&&<small>Based on source passages {quiz.result.evidence_chunk_ids.map(id=>`#${id}`).join(", ")}</small>}</details>
          {quiz.result.misconception&&<p className="learn-misconception-note">Noted a misconception to revisit: {quiz.result.misconception}</p>}
          <div className="plan-actions learn-actions"><button className="btn primary" disabled={starting!==null} onClick={()=>void startQuiz(quiz.conceptId,quiz.concept)}><RotateCcw size={14}/>Another question</button><button className="btn ghost" onClick={()=>setQuiz(null)}>Done</button></div>
        </div>}
      </div>}

      {data.due.length>0&&<div className="plan-section"><div className="plan-section-heading"><div><span className="step-number">1</span><div><h3>Review due</h3><span className="plan-help">Rate how well you remember each idea, or take a quiz.</span></div></div><span className="badge">{data.due.length}</span></div>
        <div className="learn-list">{data.due.map(item=><article className="learn-row" key={item.concept_id}>
          <div className="learn-row-main"><strong>{item.concept}</strong><small>Mastery {percent(item.mastery)} · {item.overdue_days>=1?`${Math.round(item.overdue_days)} days overdue`:"due today"}</small></div>
          <div className="learn-row-actions"><span className="learn-rate" role="group" aria-label={`Rate your recall of ${item.concept}`}>{RATINGS.map(r=><button key={r.value} className="btn outline" onClick={()=>void rate(item,r.value)}>{r.label}</button>)}</span><button className="btn primary" disabled={starting!==null} onClick={()=>void startQuiz(item.concept_id,item.concept)}>{starting===item.concept_id?<LoaderCircle size={14} className="learn-spin"/>:"Quiz me"}</button></div>
        </article>)}</div></div>}

      <div className="plan-section"><div className="plan-section-heading"><div><span className="step-number">{data.due.length>0?2:1}</span><div><h3>Up next</h3><span className="plan-help">Suggested in order. New ideas appear only after their prerequisites are solid.</span></div></div></div>
        {data.next.length?<div className="learn-list">{data.next.map(item=><article className="learn-row" key={item.concept_id}>
          <div className="learn-row-main"><strong>{item.concept}</strong><small>{item.reason}{item.blocking_prerequisites.length>0&&` · revisit first: ${item.blocking_prerequisites.join(", ")}`}</small></div>
          <div className="learn-row-actions"><button className="btn outline" disabled={starting!==null} onClick={()=>void startQuiz(item.concept_id,item.concept)}>{starting===item.concept_id?<LoaderCircle size={14} className="learn-spin"/>:<><Sparkles size={14}/>Start</>}</button></div>
        </article>)}</div>:<p className="plan-help">You're caught up. Add more sources or come back when reviews are due.</p>}
      </div>

      <div className="plan-section"><div className="plan-section-heading"><div><span className="step-number">{data.due.length>0?3:2}</span><div><h3>Mastery map</h3><span className="plan-help">Every idea found in your sources. Select one to practise it.</span></div></div></div>
        <div className="learn-map">{data.mastery.map(item=><button key={item.concept_id} className={`learn-chip ${item.status}`} disabled={starting!==null} onClick={()=>void startQuiz(item.concept_id,item.concept)} title={`${STATUS_LABEL[item.status]} · ${item.attempts} attempt${item.attempts===1?"":"s"} · ${dueLabel(item.next_due_at)}`}>
          <span>{item.concept}</span><i style={{width:percent(item.mastery)}}/><small>{STATUS_LABEL[item.status]}</small>
        </button>)}</div>
      </div>

      {data.misconceptions.length>0&&<div className="plan-section"><div className="plan-section-heading"><div><span className="step-number">!</span><div><h3>Misconceptions to fix</h3><span className="plan-help">Wrong ideas spotted in your answers. The tutor addresses these when relevant.</span></div></div></div>
        <div className="learn-list">{data.misconceptions.map(item=><article className="learn-row" key={item.id}>
          <div className="learn-row-main"><strong>{item.concept||"Unknown concept"}</strong><small>{item.description}{item.occurrences>1&&` · seen ${item.occurrences} times`}</small></div>
          <div className="learn-row-actions"><button className="btn outline" onClick={()=>void resolve(item.id)}><Check size={14}/>Resolved</button></div>
        </article>)}</div></div>}
    </>}
  </section>
}
