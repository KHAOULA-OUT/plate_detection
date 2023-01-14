import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private http:HttpClient) { }

  getPlat(path:any){
    return this.http.get("http://127.0.0.1:8000/getMatricule/"+path)
  }
  
}
