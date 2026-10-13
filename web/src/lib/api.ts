export type Course={id:number;name:string;description?:string|null;domain?:string|null;outline?:string|null;created_by?:string|null;source_count?:number}
export type SourceIssue={document_id:number;title:string;status:"needs_input"|"failed";message?:string|null}
export type Source={id:number;name:string;source_type:string;course_id?:number|null;course_name?:string|null;assignment_mode?:string;assignment_status?:string;assignment_reason?:string|null;status:string;status_message?:string|null;topic_hint?:string|null;domain_hint?:string|null;description?:string|null;failed_attempt_count?:number;document_count?:number;section_count?:number;concept_count?:number;index_pending?:number;sections_pending?:number;concepts_pending?:number;issues?:SourceIssue[];also_in?:{course_id:number;course_name:string|null}[];placement?:PlacementSuggestion|null}
// What a source's content suggests about where it belongs; the learner decides.
export type PlacementSuggestion={kind:"move"|"also"|"add";course_id:number;course_name:string|null;relation:string;confidence:number;related_source:{id:number;name:string};reason:string;leaves_space_empty:boolean}
export type PasteFormat="plain"|"srt"|"vtt"
export type PlanStatus="empty"|"ready"|"updating"|"outdated"|"failed"
export type Job={id:number;source_id:number;source_name?:string|null;source_type?:string|null;source_uri?:string|null;course_id?:number|null;status:string;stage:string;progress:string;checkpoint?:string|null;retry_count:number;created_at?:string|null;started_at?:string|null;updated_at?:string|null;error_message?:string|null}
export type MaintenanceJob={id:number;task:string;status:"queued"|"running"|"done"|"failed";progress_current:number;progress_total:number;checkpoint?:string|null;errors:Record<string,string>;updated_at?:string|null}
export type Stats={sources:number;documents:number;chunks:number;embedded:number;grouped:number;concept_indexed:number}
export type ChatTurn={role:"user"|"assistant";content:string}
export type ConversationSummary={id:string;title:string;created_at:string;updated_at:string}
export type StudyPackExport={topic:string;study_text:string;sources:{chunk_id:number;source:string;title?:string|null;section?:string|null}[];copy_paste_ready:boolean;estimated_tokens:number;max_input_tokens:number;omitted_passages:number;suggested_subtopics:string[]}
export type SessionSize="compact"|"standard"|"large"|"xl"
export type PreparedPart={id:string;title:string;purpose:string;status:"pending"|"unit"|"thin"|"split";estimated_tokens:number}
export type PreparedTopic={node_id:string;title:string;purpose:string;overlaps?:{id:string;title:string;path:string;shared:string[]}[];new_ideas?:string[];status:"unit"|"thin"|"split";depth:number;budget_tokens:number;trail:string[];next:{id:string;title:string;depth:number}|null;cached:boolean;children:PreparedPart[]|null;study_text?:string;sources?:{chunk_id:number;source:string;title?:string|null;section?:string|null;course?:string|null;is_related:boolean}[];estimated_tokens?:number;source_tokens:number;omitted_passages?:number;related_passages?:number;truncated?:boolean;note?:string|null;reason?:string;warning?:string;provisional?:boolean}
export type StudyPosition={node_id:string;title:string;trail:string[];budget_tokens:number}
export type Profile={background:string;goal:string;practice:"questions"|"coding"|"both"}
export type MasterPrompt={text:string;about_me?:boolean;estimated_tokens:number;characters:number;detailed:boolean;topics:number;completed:number;started:boolean;next_topic:{id:string;title:string;partly_done:boolean}|null;done_words:string;paths:{id:string;title:string}[];path_id:string|null}
export type MasteryStatus="unseen"|"learning"|"due"|"mastered"
export type ConceptMastery={concept_id:number;concept:string;mastery:number;confidence:number;attempts:number;status:MasteryStatus;next_due_at:string|null}
export type DueReview={concept_id:number;concept:string;mastery:number;due_at:string;overdue_days:number}
export type NextTopic={concept_id:number;concept:string;reason:string;mastery:number;blocking_prerequisites:string[]}
export type Misconception={id:number;concept_id:number;concept:string|null;description:string;occurrences:number;resolved:boolean;last_seen_at:string}
export type QuizQuestion={id:number;question:string}
export type QuizResult={quiz_item_id:number;score:number;correct:boolean;feedback:string;reference_answer:string;evidence_chunk_ids:number[];misconception:string|null;mastery:number;next_due_at:string|null}
const BASE=import.meta.env.VITE_API_URL||"http://localhost:8000"
async function req<T>(path:string,init?:RequestInit):Promise<T>{
 const r=await fetch(BASE+path,{...init,cache:"no-store"})
 if(!r.ok){let m=r.statusText;try{const d=await r.json();m=d.detail||m}catch{}throw Object.assign(new Error(m),{status:r.status})}
 return r.json()
}
export const api={
 conversations:()=>req<ConversationSummary[]>("/conversations"),
 createConversation:()=>req<{id:string;title:string}>("/conversations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:"New chat"})}),
 getConversation:(id:string)=>req<{id:string;title:string;messages:(ChatTurn&{id:string;metadata?:Record<string,any>})[]}>(`/conversations/${id}`),
 addConversationMessage:(id:string,message:ChatTurn,metadata?:Record<string,any>)=>req<any>(`/conversations/${id}/messages`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...message,metadata})}),
 deleteConversation:(id:string)=>req<any>(`/conversations/${id}`,{method:"DELETE"}),
  createCourse:(data:{name:string;domain?:string;description?:string})=>req<Course>("/courses",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}),
  // Deletes a study space; its material stays in All sources unless withMaterial.
  deleteCourse:(courseId:number,withMaterial=false)=>req<{deleted:boolean;course_id:number;removed_sources:number}>(`/courses/${courseId}?with_material=${withMaterial}`,{method:"DELETE"}),
 health:()=>req<any>("/health"),
 courses:()=>req<Course[]>("/courses"),
 sources:(courseId?:number)=>req<Source[]>(courseId?"/sources?course_id="+courseId:"/sources"),
 answerPlacement:(sourceId:number,action:"move"|"show_in_both"|"keep",courseId:number)=>req<{id:number;course_id:number|null}>(`/sources/${sourceId}/placement`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,course_id:courseId})}),
 stopShowingSource:(sourceId:number,courseId:number)=>req<{id:number;removed_from:number}>(`/sources/${sourceId}/spaces/${courseId}`,{method:"DELETE"}),
 moveSource:(sourceId:number,courseId:number|null)=>req<{id:number;course_id:number|null}>(`/sources/${sourceId}/course`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({course_id:courseId})}),
 pasteDocumentTranscript:(documentId:number,text:string,format:PasteFormat)=>req<{document_id:number;source_id:number;job_id:number}>(`/documents/${documentId}/transcript`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,input_format:format})}),
 pasteSourceText:(sourceId:number,text:string,format:PasteFormat)=>req<{source_id:number;job_id:number}>(`/sources/${sourceId}/text`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,input_format:format})}),
 reprocessSource:(sourceId:number)=>req<{source_id:number;job_id:number;documents_retried:number}>(`/sources/${sourceId}/reprocess`,{method:"POST"}),
 removeSource:(sourceId:number,failedAttemptsOnly=false)=>req<{removed:boolean;source_id:number;removed_count:number}>(`/sources/${sourceId}${failedAttemptsOnly?"?failed_attempts_only=true":""}`,{method:"DELETE"}),
 studyProgress:(courseId?:number)=>req<{completed_topics:string[];completed_topic_ids?:string[];completed_legacy_topics?:string[];partly_done_topic_ids?:string[];partly_done_budgets?:Record<string,number>;position?:StudyPosition|null}>("/study-progress"+(courseId?"?course_id="+courseId:"")),
 updateStudyProgress:(topic:string,courseId:number|undefined,completed:boolean,topicId?:string)=>req<any>("/study-progress",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,topic_id:topicId||null,course_id:courseId||null,completed})}),
 jobs:()=>req<Job[]>("/jobs"),
 organizeExisting:(includeUnassigned=true)=>req<Pick<MaintenanceJob,"id"|"status"|"progress_current"|"progress_total"|"checkpoint">&{deduplicated:boolean}>("/maintenance/organize-existing?include_unassigned="+includeUnassigned,{method:"POST"}),
 maintenanceJobs:()=>req<MaintenanceJob[]>("/maintenance/jobs"),
 // Placement checks run in the background after a source is added or moved; their suggestions appear when they finish.
 placementChecksActive:()=>req<MaintenanceJob[]>("/maintenance/jobs?task=check_placement&active=true").then(rows=>rows.length>0),
 retryMaintenanceJob:(id:number)=>req<{id:number;status:string}>(`/maintenance/jobs/${id}/retry`,{method:"POST"}),
 retryJob:(id:number)=>req<{job_id:number;source_id:number;status:string;checkpoint?:string|null}>("/jobs/"+id+"/retry",{method:"POST"}),
 stats:(id:number)=>req<Stats>("/courses/"+id+"/stats"),
 upload:async(files:File[],meta:{courseId?:number;assignmentMode?:"auto"|"explicit_course"|"explicit_unassigned";topic?:string;domain?:string;description?:string})=>{
  const out:any[]=[]
  for(const file of files){
   const f=new FormData();f.append("file",file)
   if(meta.courseId)f.append("course_id",String(meta.courseId))
   f.append("assignment_mode",meta.assignmentMode||(meta.courseId?"explicit_course":"auto"))
   if(meta.topic)f.append("topic_hint",meta.topic)
   if(meta.domain)f.append("domain_hint",meta.domain)
   if(meta.description)f.append("description",meta.description)
   out.push(await req<{deduplicated?:boolean;assignment_conflict?:boolean}>("/sources/file",{method:"POST",body:f}))
  }
  return out
 },
 manual:(data:FormData)=>req("/sources/manual",{method:"POST",body:data}),
 youtube:(url:string,meta:{courseId?:number;assignmentMode?:"auto"|"explicit_course"|"explicit_unassigned";name?:string;topic?:string;domain?:string;description?:string})=>{
  const q=new URLSearchParams({url})
  if(meta.courseId)q.set("course_id",String(meta.courseId))
  q.set("assignment_mode",meta.assignmentMode||(meta.courseId?"explicit_course":"auto"))
  if(meta.name)q.set("name",meta.name)
  if(meta.topic)q.set("topic_hint",meta.topic)
  if(meta.domain)q.set("domain_hint",meta.domain)
  if(meta.description)q.set("description",meta.description)
  return req<{source_id:number;job_id:number|null;status:string;deduplicated?:boolean;assignment_conflict?:boolean;course_id?:number|null}>("/sources/youtube?"+q.toString(),{method:"POST"})
 },
 syllabus:(id?:number,regenerate=false)=>req<{syllabus:string;revision:string;source_relationships:{sources:any[];cross_paths:any[];shared_foundations:string[]};nodes:{id:string;parent_id:string|null;path_id:string;type:"outline"|"study";title:string;metadata:Record<string,any>}[];warnings?:string[];status?:PlanStatus;error?:string|null;up_to_date?:boolean;generated_at?:string|null}>("/syllabus"+(id||regenerate?"?"+new URLSearchParams({...(id?{course_id:String(id)}:{}),...(regenerate?{regenerate:"true"}:{})}).toString():"")),
 studyPack:(topic:string,id?:number)=>req<any>("/study-pack",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,course_id:id||null,top_k:12})}),
 // retry=false only shows a topic again: a result made while the AI was unavailable is not prepared again.
 prepareTopic:(nodeId:string,courseId:number|undefined,size:SessionSize,tutor:string,retry=true)=>req<PreparedTopic>("/plan/topics/prepare",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({node_id:nodeId,course_id:courseId||null,size,tutor,retry})}),
 profile:()=>req<Profile>("/profile"),
 saveProfile:(data:Profile)=>req<Profile>("/profile",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}),
 masterPrompt:(courseId:number|undefined,tutor:string,pathId?:string)=>req<MasterPrompt>("/plan/master-prompt?"+new URLSearchParams({...(courseId?{course_id:String(courseId)}:{}),tutor,...(pathId?{path_id:pathId}:{})}).toString()),
 exportStudyPack:(topic:string,id:number|undefined,tutor:string,maxInputTokens:number,topicId?:string)=>req<StudyPackExport>("/study-pack/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,topic_id:topicId||null,course_id:id||null,top_k:20,tutor,max_input_tokens:maxInputTokens})}),
 study:(question:string,id?:number,history:ChatTurn[]=[],retrievalScope:"space_only"|"space_and_related"|"all_sources"="space_only")=>req<any>("/study",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question,course_id:id||null,top_k:8,history:history.slice(-8),retrieval_scope:retrievalScope})}),
 learnerMastery:(courseId?:number)=>req<{concepts:ConceptMastery[]}>("/learner/mastery"+(courseId?"?course_id="+courseId:"")),
 learnerDue:(courseId?:number)=>req<{due:DueReview[]}>("/learner/due"+(courseId?"?course_id="+courseId:"")),
 learnerNext:(courseId?:number)=>req<{next:NextTopic[]}>("/learner/next"+(courseId?"?course_id="+courseId:"")),
 misconceptions:()=>req<{misconceptions:Misconception[]}>("/learner/misconceptions"),
 resolveMisconception:(id:number)=>req<{id:number;resolved:boolean}>(`/learner/misconceptions/${id}/resolve`,{method:"POST"}),
 selfRating:(conceptId:number,rating:number,courseId?:number)=>req<{concept_id:number;mastery:number;next_due_at:string|null}>("/learner/self-rating",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({concept_id:conceptId,rating,course_id:courseId||null})}),
 generateQuiz:(conceptId:number,courseId?:number)=>req<{concept_id:number;concept:string;items:QuizQuestion[]}>("/quiz/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({concept_id:conceptId,course_id:courseId||null,count:1})}),
 answerQuiz:(itemId:number,answer:string)=>req<QuizResult>(`/quiz/${itemId}/answer`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({answer})})
}
