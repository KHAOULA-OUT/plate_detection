import { Component } from '@angular/core';
import { ApiService } from './services/api.service';
class ImageSnippet {
  constructor(public src: string, public file: File) {}
}
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'matricule';
  nb_plate:any;
  path:any;
  path_angular:any;
  path_contour:any;
  path_countour_rect:any;
  isReload:any;
  
  selectedFile: ImageSnippet | undefined;
  constructor(private api:ApiService){}

  ngOnInit(){
    this.isReload=false;
  }
  reload(){
    this.path_contour=null;
    this.path_countour_rect=null;
    this.path="";
    this.path_angular="";
    window.location.reload();
  }
   async getPlat(){
    this.api.getPlat(this.path).subscribe((res:any)=>{
      if(res.matricule=="Detetion failed, please Try again"){
        this.nb_plate=res.matricule;
      }else{
        this.nb_plate=res.matricule;
        this.path_contour="./assets/images/contour.jpg";
        this.path_countour_rect="./assets/images/countour_rect.jpg";
      }
      
      this.isReload=true;
    })
  }

  async enterPath(pt: any){
    this.path=pt;
  }
  async onSelectImage(event:any) {
    if (event.target.files){
      await this.enterPath(event.target.files[0].name);
      this.path_angular="./assets/images/"+event.target.files[0].name;      
      console.log(event.target.files[0].name);
    }
  }

  
}
