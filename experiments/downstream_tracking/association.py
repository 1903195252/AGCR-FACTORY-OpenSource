"""Class-compatible association; unavailable geometry uses appearance only."""
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT=dict(confidence=.5,appearance_weight=.6,appearance_gate=.7,mahalanobis_gate=16.27,
             process_variance=.5,measurement_variance=.25,max_age_seconds=5.)


class CooperativeTracker:
    def __init__(self,parameters=None):
        self.config=dict(DEFAULT,**(parameters or {}));self.tracks={};self.next_id=1;self.last_time=None

    def _predict(self,time_s):
        if self.last_time is not None:
            dt=max(0,float(time_s)-self.last_time);F=np.eye(6);F[:3,3:]=np.eye(3)*dt
            Q=np.eye(6)*self.config['process_variance']*max(dt,1e-3)
            for track in self.tracks.values():
                if track['mean'] is None:continue
                track['mean']=F@track['mean'];track['cov']=F@track['cov']@F.T+Q
        self.last_time=float(time_s)
        self.tracks={k:t for k,t in self.tracks.items() if time_s-t['seen']<=self.config['max_age_seconds']}

    def _measure(self,d):
        # The RGB ray/floor intersection is a diagnostic proxy, not a metric
        # center measurement with calibrated covariance. Keep the raw fields
        # on detections, but never let this proxy gate or update world tracks.
        if d.get('localization_scope')=='floor_projection':return None
        if d.get('position_world_m') is None:return None
        z=np.asarray(d['position_world_m'],float)
        if z.shape!=(3,) or not np.isfinite(z).all():return None
        R=np.asarray(d.get('center_covariance_world_m2',np.eye(3)*self.config['measurement_variance']),float)
        if R.shape!=(3,3) or not np.isfinite(R).all():return None
        return z,R+np.eye(3)*1e-4

    def _new(self,d,time_s):
        measurement=self._measure(d);k=self.next_id;self.next_id+=1
        mean=cov=None
        if measurement is not None:
            z,R=measurement;mean=np.r_[z,np.zeros(3)];cov=np.eye(6)*4;cov[:3,:3]=R
        self.tracks[k]=dict(mean=mean,cov=cov,descriptor=np.asarray(d['identity_embedding'],float),
                            class_id=d['class_id'],seen=float(time_s))
        return k

    def _cost(self,detections,keys):
        cost=np.full((len(detections),len(keys)),1e6,float)
        if not detections or not keys:return cost
        classes={d['class_id'] for d in detections}
        for category in classes:
            rows=[i for i,d in enumerate(detections) if d['class_id']==category]
            columns=[j for j,k in enumerate(keys) if self.tracks[k]['class_id']==category]
            if not columns:continue
            tracks=[self.tracks[keys[j]] for j in columns]
            descriptors=np.stack([t['descriptor'] for t in tracks])
            for start in range(0,len(rows),32):
                selected=rows[start:start+32]
                measurements=[self._measure(detections[i]) for i in selected]
                embeddings=np.stack([detections[i]['identity_embedding'] for i in selected])
                appearance=1-np.einsum('id,jd->ij',embeddings,descriptors)
                weight=self.config['appearance_weight']
                # Missing geometry is not a zero-distance measurement. Preserve
                # the fixed appearance gate and use its appearance-only cost.
                value=appearance.copy();gate=appearance<=self.config['appearance_gate']
                mr=[i for i,m in enumerate(measurements) if m is not None]
                mc=[j for j,t in enumerate(tracks) if t['mean'] is not None]
                if mr and mc:
                    z=np.stack([measurements[i][0] for i in mr]);R=np.stack([measurements[i][1] for i in mr])
                    means=np.stack([tracks[j]['mean'][:3] for j in mc]);cov=np.stack([tracks[j]['cov'][:3,:3] for j in mc])
                    delta=z[:,None]-means[None]
                    solved=np.linalg.solve(cov[None]+R[:,None],delta[...,None])[...,0]
                    maha=np.einsum('...i,...i->...',delta,solved);ix=np.ix_(mr,mc)
                    gate[ix]&=maha<=self.config['mahalanobis_gate']
                    value[ix]=weight*appearance[ix]+(1-weight)*maha/self.config['mahalanobis_gate']
                cost[np.ix_(selected,columns)]=np.where(gate,value,1e6)
        return cost

    def update(self,paired,time_s):
        """paired={air:[detections],ground:[detections]}; mutates no inputs."""
        self._predict(time_s);result={}
        for view in ('air','ground'):
            detections=[dict(d) for d in paired.get(view,[]) if d['score']>=self.config['confidence']]
            keys=list(self.tracks);cost=np.full((len(detections),len(keys)),1e6,float)
            cost=self._cost(detections,keys)
            assigned={}
            if cost.size:
                a,b=linear_sum_assignment(cost)
                assigned={i:keys[j] for i,j in zip(a,b) if cost[i,j]<1e5}
            for i,d in enumerate(detections):
                if i not in assigned:k=self._new(d,time_s)
                else:
                    k=assigned[i];tr=self.tracks[k];measurement=self._measure(d)
                    if measurement is not None:
                        z,R=measurement
                        if tr['mean'] is None:
                            tr['mean']=np.r_[z,np.zeros(3)];tr['cov']=np.eye(6)*4;tr['cov'][:3,:3]=R
                        else:
                            H=np.eye(3,6);gain=tr['cov']@H.T@np.linalg.inv(H@tr['cov']@H.T+R)
                            tr['mean']+=gain@(z-H@tr['mean']);A=np.eye(6)-gain@H
                            tr['cov']=A@tr['cov']@A.T+gain@R@gain.T
                    tr['descriptor']=.9*tr['descriptor']+.1*np.asarray(d['identity_embedding'])
                    tr['descriptor']/=max(np.linalg.norm(tr['descriptor']),1e-9);tr['seen']=float(time_s)
                tr=self.tracks[k];d.update(track_id=int(k),velocity_world_mps=None if tr['mean'] is None else tr['mean'][3:].tolist(),
                    filtered_position_world_m=None if tr['mean'] is None else tr['mean'][:3].tolist(),
                    association_geometry_available=self._measure(d) is not None)
            result[view]=detections
        return result
