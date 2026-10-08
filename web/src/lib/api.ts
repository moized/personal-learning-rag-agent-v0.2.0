export type Course={id:number;name:string;description?:string|null;domain?:string|null;outline?:string|null}
export type Source={id:number;name:string;source_type:string;course_id?:number|null;assignment_mode?:string;assignment_status?:string;status:string;topic_hint?:string|null;domain_hint?:string|null;description?:string|null;failed_attempt_count?:number}
export type Job={id:number;source_id:number;source_name?:string|null;source_type?:string|null;source_uri?:string|null;course_id?:number|null;status:string;stage:string;progress:string;checkpoint?:string|null;retry_count:number;created_at?:string|null;started_at?:string|null;updated_at?:string|null;error_message?:string|null}
export type MaintenanceJob={id:number;task:string;status:"queued"|"running"|"done"|"failed";progress_current:number;progress_total:number;checkpoint?:string|null;errors:Record<string,string>;updated_at?:string|null}
export type Stats={sources:number;documents:number;chunks:number;embedded:number;grouped:number;concept_indexed:number}
export type ChatTurn={role:"user"|"assistant";content:string}
export type ConversationSummary={id:string;title:string;created_at:string;updated_at:string}
export type StudyPackExport={topic:string;study_text:string;sources:{chunk_id:number;source:string;title?:string|null;section?:string|null}[];copy_paste_ready:boolean;estimated_tokens:number;max_input_tokens:number;omitted_passages:number;suggested_subtopics:string[]}
const BASE=import.meta.env.VITE_API_URL||"http://localhost:8000"
async function req<T>(path:string,init?:RequestInit):Promise<T>{
 const r=await fetch(BASE+path,{...init,cache:"no-store"})
 if(!r.ok){let m=r.statusText;try{const d=await r.json();m=d.detail||m}catch{}throw new Error(m)}
 return r.json()
}
export const api={
 conversations:()=>req<ConversationSummary[]>("/conversations"),
 createConversation:()=>req<{id:string;title:string}>("/conversations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:"New chat"})}),
 getConversation:(id:string)=>req<{id:string;title:string;messages:(ChatTurn&{id:string;metadata?:Record<string,any>})[]}>(`/conversations/${id}`),
 addConversationMessage:(id:string,message:ChatTurn,metadata?:Record<string,any>)=>req<any>(`/conversations/${id}/messages`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...message,metadata})}),
 deleteConversation:(id:string)=>req<any>(`/conversations/${id}`,{method:"DELETE"}),
  createCourse:(data:{name:string;domain?:string;description?:string})=>req<Course>("/courses",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}),
 health:()=>req<any>("/health"),
 courses:()=>req<Course[]>("/courses"),
 sources:(courseId?:number)=>req<Source[]>(courseId?"/sources?course_id="+courseId:"/sources"),
 moveSource:(sourceId:number,courseId:number|null)=>req<{id:number;course_id:number|null}>(`/sources/${sourceId}/course`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({course_id:courseId})}),
 removeSource:(sourceId:number,failedAttemptsOnly=false)=>req<{removed:boolean;source_id:number;removed_count:number}>(`/sources/${sourceId}${failedAttemptsOnly?"?failed_attempts_only=true":""}`,{method:"DELETE"}),
 studyProgress:(courseId?:number)=>req<{completed_topics:string[]}>("/study-progress"+(courseId?"?course_id="+courseId:"")),
 updateStudyProgress:(topic:string,courseId:number|undefined,completed:boolean)=>req<any>("/study-progress",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,course_id:courseId||null,completed})}),
 jobs:()=>req<Job[]>("/jobs"),
 organizeExisting:(includeUnassigned=true)=>req<Pick<MaintenanceJob,"id"|"status"|"progress_current"|"progress_total"|"checkpoint">&{deduplicated:boolean}>("/maintenance/organize-existing?include_unassigned="+includeUnassigned,{method:"POST"}),
 maintenanceJobs:()=>req<MaintenanceJob[]>("/maintenance/jobs"),
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
 syllabus:(id?:number)=>req<{syllabus:string;revision:string;source_relationships:{sources:any[];cross_paths:any[];shared_foundations:string[]};nodes:{id:string;parent_id:string|null;path_id:string;type:"outline"|"study";title:string;metadata:Record<string,any>}[]}>("/syllabus"+(id?"?course_id="+id:"")),
 studyPack:(topic:string,id?:number)=>req<any>("/study-pack",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,course_id:id||null,top_k:12})}),
 exportStudyPack:(topic:string,id:number|undefined,tutor:string,maxInputTokens:number,topicId?:string)=>req<StudyPackExport>("/study-pack/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,topic_id:topicId||null,course_id:id||null,top_k:20,tutor,max_input_tokens:maxInputTokens})}),
 study:(question:string,id?:number,history:ChatTurn[]=[],retrievalScope:"space_only"|"space_and_related"|"all_sources"="space_only")=>req<any>("/study",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question,course_id:id||null,top_k:8,history:history.slice(-8),retrieval_scope:retrievalScope})})
}
