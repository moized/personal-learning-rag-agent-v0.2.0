export type Course={id:number;name:string;description?:string|null;domain?:string|null;outline?:string|null}
export type Source={id:number;name:string;source_type:string;course_id?:number|null;status:string;topic_hint?:string|null;domain_hint?:string|null;description?:string|null}
export type Job={id:number;source_id:number;source_name?:string|null;course_id?:number|null;status:string;stage:string;progress:string;checkpoint?:string|null;error?:string|null}
export type Stats={sources:number;documents:number;chunks:number;embedded:number;grouped:number;concept_indexed:number}
const BASE=import.meta.env.VITE_API_URL||"http://localhost:8000"
async function req<T>(path:string,init?:RequestInit):Promise<T>{
 const r=await fetch(BASE+path,{...init,cache:"no-store"})
 if(!r.ok){let m=r.statusText;try{const d=await r.json();m=d.detail||m}catch{}throw new Error(m)}
 return r.json()
}
export const api={
  createCourse:(data:{name:string;domain?:string;description?:string})=>req("/courses",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}),
 health:()=>req<any>("/health"),
 courses:()=>req<Course[]>("/courses"),
 sources:(courseId?:number)=>req<Source[]>(courseId?"/sources?course_id="+courseId:"/sources"),
 jobs:()=>req<Job[]>("/jobs"),
 stats:(id:number)=>req<Stats>("/courses/"+id+"/stats"),
 upload:async(files:File[],meta:{courseId?:number;topic?:string;domain?:string;description?:string})=>{
  const out:any[]=[]
  for(const file of files){
   const f=new FormData();f.append("file",file)
   if(meta.courseId)f.append("course_id",String(meta.courseId))
   if(meta.topic)f.append("topic_hint",meta.topic)
   if(meta.domain)f.append("domain_hint",meta.domain)
   if(meta.description)f.append("description",meta.description)
   out.push(await req("/sources/file",{method:"POST",body:f}))
  }
  return out
 },
 manual:(data:FormData)=>req("/sources/manual",{method:"POST",body:data}),
 youtube:(url:string,meta:{courseId?:number;name?:string;topic?:string;domain?:string;description?:string})=>{
  const q=new URLSearchParams({url})
  if(meta.courseId)q.set("course_id",String(meta.courseId))
  if(meta.name)q.set("name",meta.name)
  if(meta.topic)q.set("topic_hint",meta.topic)
  if(meta.domain)q.set("domain_hint",meta.domain)
  if(meta.description)q.set("description",meta.description)
  return req("/sources/youtube?"+q.toString(),{method:"POST"})
 },
 syllabus:(id?:number)=>req<{syllabus:string}>("/syllabus"+(id?"?course_id="+id:"")),
 studyPack:(topic:string,id?:number)=>req<any>("/study-pack",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({topic,course_id:id||null,top_k:12})}),
 study:(question:string,id?:number)=>req<any>("/study",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question,course_id:id||null,top_k:8})})
}