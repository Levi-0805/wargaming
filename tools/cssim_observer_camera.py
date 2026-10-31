"""Read-only director snapshots using OCR-verified unit-list focus.

Clicks unit names for viewing only. Never touches AI/manual/task controls.
Windows OCR is local and uses no model/API credits. Unknown UI means skip.
"""
from pathlib import Path
import json,math,re,subprocess,sys,time
from collections import Counter
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists() and (p/'PythonCode').is_dir())
sys.path.insert(0,str(ROOT/'tools'))
from cssim_desktop import Desktop
from cssim_ui_flow import read_image
from cssim_campaign import WindowsProcesses,same_process
import batch_training as files
import cv2
from PIL import Image
GROUND={'BP_BaseSoldier_C','BP_RoboDog_C','BP_MNWS_Vehicle_Armored_C','BP_MNWS_Vehicle_6x6UGV_C'}


def normalize(text):
    text=re.sub(r'\s+','',str(text))
    text=text.replace('十兵','士兵').replace('土兵','士兵').replace('±兵','士兵')
    text=text.replace(']','1').replace('丨','1').replace('|','1').replace('一','')
    return re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]','',text)


class Viewer:
    def __init__(self,output):
        self.out=Path(output);self.desktop=Desktop();self.sets=Counter();self.last_shot=0.;self.last_attempt=0.;self.sequence=0
        assets=self.out/'assets';assets.mkdir(exist_ok=True)
        self.templates={}
        for name in ('red','director','panel','expanded','collapsed'):
            source=Path(__file__).with_name('observer_assets')/(name+'.png')
            target=assets/(name+'.png')
            if not target.exists() or files.sha(target)!=files.sha(source):
                import shutil
                shutil.copyfile(source,target)
            self.templates[name]=read_image(target)

    def score(self,image,name,box):
        x1,y1,x2,y2=box;roi=image[y1:y2,x1:x2]
        return cv2.minMaxLoc(cv2.matchTemplate(roi,self.templates[name],cv2.TM_CCOEFF_NORMED))[1]

    def photograph(self,hwnd,name):return self.desktop.capture(hwnd,self.out/name)

    def ocr_rows(self,photo):
        image=self.out/'scratch/list-ocr.png';output=self.out/'scratch/list-ocr.json'
        Image.open(photo['image_path']).crop((112,214,253,640)).resize((423,1278)).save(image)
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden',
            '-File',str(Path(__file__).with_name('cssim_observer_ocr.ps1')),'-Image',str(image.resolve()),'-Out',str(output.resolve())],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
        return files.read_json(output)

    def wheel(self,hwnd,count):
        n=self.desktop.backend;n.ensure_foreground(hwnd)
        for _ in range(abs(count)):
            e=n.INPUT(type=0);e.mi.dwFlags=0x0800;e.mi.mouseData=(120 if count>0 else -120)&0xffffffff
            n._send([e]);time.sleep(.09)
        time.sleep(.7)

    def pitch(self,hwnd):
        # Relative mouse input is required by Unreal; SetCursorPos alone does not rotate its camera.
        n=self.desktop.backend;w=n.describe(hwnd);n.move_screen(hwnd,w['client_rect'][0]+800,w['client_rect'][1]+400)
        e=n.INPUT(type=0);e.mi.dwFlags=8;n._send([e])
        try:
            for dy,count in ((10,60),(-5,20)):
                for _ in range(count):
                    e=n.INPUT(type=0);e.mi.dwFlags=1;e.mi.dy=dy;n._send([e]);time.sleep(.02)
        finally:
            e=n.INPUT(type=0);e.mi.dwFlags=16;n._send([e])

    def focus(self,hwnd,engine):
        current=self.photograph(hwnd,'scratch/current.png');image=read_image(current['image_path'])
        if self.score(image,'panel',(65,115,220,155))<.93:self.desktop.click(current,34,323)
        current=self.photograph(hwnd,'scratch/current.png')
        if self.score(read_image(current['image_path']),'panel',(65,115,220,155))<.93:return None
        self.desktop.move(current,210,550);self.wheel(hwnd,18)
        current=self.photograph(hwnd,'scratch/current.png')
        self.desktop.click(current,337,163)  # Clear a viewing filter; no scenario/algorithm setting changes.
        current=self.photograph(hwnd,'scratch/current.png');image=read_image(current['image_path'])
        expanded=self.score(image,'expanded',(86,201,97,212));collapsed=self.score(image,'collapsed',(86,201,97,212))
        if collapsed>expanded+.1:
            self.desktop.click(current,92,207);current=self.photograph(hwnd,'scratch/current.png')
        elif expanded<.8:return None
        available={normalize(u.get('label','')):u for u in engine.units.values()
            if u.get('team')==0 and u.get('alive') and u['type'] in GROUND and u.get('label')}
        candidates=[]
        for page in range(2):
            if page:
                self.desktop.move(current,210,550);self.wheel(hwnd,-12)
                current=self.photograph(hwnd,'scratch/current.png')
            for line in self.ocr_rows(current):
                words=line.get('words') or []
                if not words:continue
                text=normalize(line['text'])
                for name,unit in available.items():
                    if name and (text==name or text.endswith(name) and len(text)<=len(name)+2):
                        y=214+(min(w['y'] for w in words)+max(w['y']+w['height'] for w in words))/6
                        candidates.append((unit,round(y)))
            if candidates:break
        if not candidates:return None
        preferred=(engine.last_live_event or {}).get('detail',{}).get('uid')
        def priority(pair):
            u,_=pair;sample=u.get('protocol_sample') or {};contacts=sample.get('local_contacts') or []
            return (u['uid']!=preferred,not bool(contacts),'Vehicle' not in u['type'],u['uid'])
        unit,y=min(candidates,key=priority)
        if not 214<y<640:return None
        _,window=self.desktop._action_window(current);n=self.desktop.backend
        n.move_screen(hwnd,window['client_rect'][0]+180,window['client_rect'][1]+y)
        for _ in range(2):
            e=n.INPUT(type=0);e.mi.dwFlags=2;n._send([e]);time.sleep(.05)
            e=n.INPUT(type=0);e.mi.dwFlags=4;n._send([e]);time.sleep(.12)
        time.sleep(6)
        current=self.photograph(hwnd,'scratch/current.png');self.desktop.key(current,'Escape')
        current=self.photograph(hwnd,'scratch/current.png')
        if self.score(read_image(current['image_path']),'panel',(65,115,220,155))>.93:
            self.desktop.click(current,34,323)
        self.pitch(hwnd);self.wheel(hwnd,-18)
        return dict(uid=unit['uid'],label=unit['label'],reference_position=unit['position'],
                    selection_evidence='Exact normalized local OCR match in unit-name column; name double-click only')

    def capture(self,learning,campaign,engine,*,force=False):
        if (self.out/'VIEW_PAUSE').exists() or not engine.frame or campaign.get('state')!='collecting':return None
        if not 8<engine.frame['step']<465:return None
        key=(engine.frame['batch_id'],engine.frame['episode']);urgent=time.monotonic()<engine.urgent_until
        if self.sets[key]>=(6 if urgent else 2):return None
        if not force and time.monotonic()-self.last_shot<(60 if urgent else 600):return None
        if time.monotonic()-self.last_attempt<(30 if urgent else 90):return None
        self.last_attempt=time.monotonic()
        attempt=ROOT/'logs/campaigns'/learning['active_campaign']/('attempt-'+campaign['current_attempt'].rsplit('-',1)[1])
        owner=next((p for p in files.read_json(attempt/'processes.json')['owned'] if p['role']=='client'),None)
        if not owner or not same_process(owner,WindowsProcesses().identity(owner['pid'])):return None
        windows=[w for w in self.desktop.windows() if w['pid']==owner['pid'] and
                 w['client_rect'][2]-w['client_rect'][0]==1600 and w['client_rect'][3]-w['client_rect'][1]==900]
        if len(windows)!=1:return None
        hwnd=windows[0]['hwnd'];photo=self.photograph(hwnd,'scratch/current.png');self.desktop.move(photo,1500,700)
        photo=self.photograph(hwnd,'scratch/current.png');image=read_image(photo['image_path'])
        if max(self.score(image,'red',(760,100,842,135)),self.score(image,'director',(760,100,842,135)))<.93:return None
        if self.score(image,'director',(760,100,842,135))<.93:
            self.desktop.click(photo,800,117);photo=self.photograph(hwnd,'scratch/current.png');self.desktop.move(photo,1500,700)
            photo=self.photograph(hwnd,'scratch/current.png')
        if self.score(read_image(photo['image_path']),'director',(760,100,842,135))<.93:return None
        self.sequence+=1;stamp=f'{int(time.time())}-{self.sequence:03d}'
        before=self.photograph(hwnd,'views/'+stamp+'-director-context.png')
        focused=self.focus(hwnd,engine)
        if not focused:
            photo=self.photograph(hwnd,'scratch/current.png')
            if self.score(read_image(photo['image_path']),'panel',(65,115,220,155))>.93:self.desktop.click(photo,34,323)
            files.write_json(self.out/'view-skip.json',dict(at=files.now(),reason='No OCR-verified live ground unit visible; skip low-value close-up',frame=engine.frame))
            return None
        close=self.photograph(hwnd,'views/'+stamp+'-director-focused.png')
        self.wheel(hwnd,-4)
        overview=self.photograph(hwnd,'views/'+stamp+'-director-area.png')
        self.sets[key]+=1;self.last_shot=time.monotonic()
        context=dict(captured_at=files.now(),reference_log_frame=engine.frame,frame=engine.frame,
            images=[before['image_path'],close['image_path'],overview['image_path']],director_verified=True,
            view_quality='Requires screenshot HUD/scale review; focus request does not prove a close-up',
            focused_unit=focused,view_left_as='Battle-area overview; original camera transform is not asserted restored',
            event=engine.last_live_event,source_commit=engine.context['source_commit'],
            limitations=['UI screenshots may be several steps later than reference log frame; compare screenshot HUD.',
                        'Building roofs may occlude figures; HP/ammo/actions are checked in corresponding logs.',
                        'Viewer-only focus and OCR; no AI/manual/task-control menu action is sent.'])
        files.write_json(self.out/'views'/f'{stamp}-context.json',context);files.write_json(self.out/'latest-view.json',context)
        return context
