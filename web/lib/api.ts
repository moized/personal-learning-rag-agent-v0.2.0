export type Course={id:number;name:string;description?:string|null;domain?:string|null;outline?:string|null}
export type Source={id:number;name:string;source_type:string;course_id?:number|null;status:string;topic_hint?:string|null;domain_hint?:string|null;description?:string|null}
export type Job={id:number;source_id:number;source_name?:string|null;status:string;stage:string;progress:string;checkpoint?:string|null;error?:string|null}
export type CourseStats={course_id:number;sources:number;documents:number;chunks:number;embedded:number;grouped:number;concept_indexed:number}
export type StudyResponse={answer:string;sources:Array<{chunk_id:number;score:number;text:string;source:string;document?:string|null}>;selected_group_ids?:number[]}
export type StudyPackResponse={topic:string;study_text:string;sources:Array<{chunk_id:number;score:number;source:string;document?:string|null}>;copy_paste_ready:boolean}
const base=process.env.NEXT_PUBLIC_API_URL??"http://localhost:8000"
async function request<T>(path:string,init?:RequestInit):Promise<T>{const r=await fetch(base+path,{...init,headers:{...(init?.headers??{})},cache:"no-store"});if(!r.ok){let d=r.statusText;try{const p=await r.json();d=p.detail??d}catch{}throw new Error(d)}return r.json()}
export const api={
 health:()=>request<{status:string;llm:boolean;reasoning_model:string;fast_model:string;embedding_model:string}>("/health"),
 courses:()=>request<Course[]>("/courses"),
 sources:(id?:number)=>request<Source[]>(`/sources${id?`?course_id=${id}`:""}`),
 jobs:()=>request<Job[]>("/jobs"),
 stats:(id:number)=>request<CourseStats>(`/courses/${id}/stats`),
 uploadFiles:async(files:File[],m:{courseId?:number;topic?:string;domain?:string;description?:string})=>{const out=[];for(const file of files){const f=new FormData();f.append("file",file);if(m.courseId)f.append("course_id",String(m.courseId));if(m.topic)f.append("topic_hint",m.topic);if(m.domain)f.append("domain_hint",m.domain);if(m.description)f.append("description",m.description);out.push(await request("/sources/file",{method:"POST",body:f}))}return out},
 addManual:(f:FormData)=>request("/sources/manual",{method:"POST",body:f}),
 addYoutube:(url:string,m:{name?:string;courseId?:number;topic?:string;domain?:string;description?:string})=>{const p=new URLSearchParams({url});if(m.name)p.set("name",m.name);if(m.courseId)p.set("course_id",String(m.courseId));if(m.topic)p.set("topic_hint",m.topic);if(m.domain)p.set("domain_hint",m.domain);if(m.description)p.set("description",m.description);return request(`/sources/youtube?${p}`,{method:"POST"})},
 syllabus:(id?:number)=>request<{syllabus:string}>(`/syllabus${id?`?course_id=${id}`:""}`),
 studyPack:(topic:string,id?:number)=>request<StudyPackResponse>("/study-pack",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,course_id:id??null,top_k:12})}),
 study:(question:string,id?:number)=>request<StudyResponse>("/study",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question,course_id:id??null,top_k:8})})
}