"""All-time residual enclosure using one M-matrix ground-state supersolution.

Exact-arithmetic theorem; ordinary floating-point implementation. No FOM time
integration, spectrum, inverse residual actions, or temporal sampling acceptance.
"""
import math
import time
import numpy as np
import scipy.linalg as la
from numerics import sym, f


class DiagonalTimeCertificate:
    def __init__(self, K, C, H, G, V, D, alpha, time_ratio=1.1):
        started = time.perf_counter()
        if np.any(D <= 0) or alpha <= 0 or time_ratio <= 1:
            raise ValueError('invalid diagonal certificate or time ratio')
        self.alpha, self.ratio = alpha, time_ratio
        self.r = V.shape[1]
        mass = sym(V.T @ (C @ V))
        # Caller uses a C-orthonormal basis; do not silently assume this.
        if la.norm(mass-np.eye(self.r),2) > 1e-8:
            raise ValueError('C-orthonormal basis required')
        self.Ar = np.asarray([sym(V.T @ (A @ V)) for A in [K]+H])
        self.F = V.T @ G
        q0 = sym(G.T @ (G/C.diagonal()[:,None]))
        l,Z = la.eigh(q0)
        if l[0] <= 0:
            raise ValueError('dependent input columns')
        self.Z = Z/np.sqrt(l)
        self.Fz = self.F @ self.Z
        CV = C @ V
        R0 = G-CV@self.F
        blocks = [R0] + [CV@ar-A@V for A,ar in zip([K]+H,self.Ar)]
        R = np.column_stack(blocks)
        # Weighted QR avoids squared-Gram cancellation near exact residuals.
        self.Tc = la.qr(R/np.sqrt(C.diagonal())[:,None],mode='economic',check_finite=False)[1]
        self.Td = la.qr(R/np.sqrt(D)[:,None],mode='economic',check_finite=False)[1]
        self.build_seconds = time.perf_counter()-started

    def _coordinates(self,T,h,U):
        p=self.F.shape[1]
        J=T[:,p:p+self.r].copy()
        for i,x in enumerate(h):
            J += x*T[:,p+(i+1)*self.r:p+(i+2)*self.r]
        return T[:,:p]@self.Z, J@U

    def evaluate(self,h,epsilon=.001,early_reject=True):
        start=time.perf_counter()
        A=self.Ar[0]+np.einsum('i,ijk->jk',h,self.Ar[1:])
        lam,U=la.eigh(sym(A))
        if lam[0] <= 0:
            return dict(passed=False,reason='nonpositive reduced rate')
        F=U.T@self.Fz
        Rc,Jc=self._coordinates(self.Tc,h,U)
        Rd,Jd=self._coordinates(self.Td,h,U)
        # Steady certificate is Galerkin energy best-approximation, including
        # arbitrary signed input combinations and the reduced denominator.
        rss=Rd+Jd@(F/lam[:,None])
        q=sym(F.T@(F/lam[:,None]))
        steady2=max(0.,float(la.eigvalsh(sym(rss.T@rss),q)[-1]))
        steady=np.sqrt(steady2/(1+steady2))
        if early_reject and steady > epsilon:
            return dict(passed=False,steady_bound=float(steady),step_bound=None,
                        reason='steady enclosure failed',seconds=time.perf_counter()-start)
        nc=la.norm(Jc,axis=0)*la.norm(F,axis=1)
        nd=la.norm(Jd,axis=0)*la.norm(F,axis=1)
        r0=la.norm(Rc,2)
        t0=1e-7/lam[-1]
        def den(t):
            return la.svdvals(f(t,lam)[:,None]*F)[-1]
        ec=t0*r0+t0*t0*sum(nc)/2
        worst=ec/den(t0)
        intervals=0
        tail_time=40/lam[0]
        target=np.sqrt(epsilon)/(1+np.sqrt(epsilon))
        edge_count=math.ceil(math.log(tail_time/t0)/math.log(self.ratio))
        edges=np.r_[t0*self.ratio**np.arange(edge_count),tail_time]
        aa,bb=edges[:-1],edges[1:]
        mm=(aa+bb)/2; widths=bb-aa
        rates=np.exp(-mm[:,None]*lam)
        def modal_fields(J,values):
            coefficients=(values[:,:,None]*F[None,:,:]).transpose(1,0,2).reshape(len(lam),-1)
            return (J@coefficients).reshape(J.shape[0],len(mm),F.shape[1]).transpose(1,0,2)
        def norms(fields):
            return np.linalg.svd(fields,compute_uv=False)[:,0]
        fm=-np.expm1(-mm[:,None]*lam)/lam
        rc=norms(modal_fields(Jc,fm)+Rc[None,:,:])
        rd=norms(modal_fields(Jd,fm)+Rd[None,:,:])
        for k in range(1,4):
            values=rates*((-lam)**(k-1))
            scale=(widths/2)**k/math.factorial(k)
            rc += scale*norms(modal_fields(Jc,values))
            rd += scale*norms(modal_fields(Jd,values))
        remainder=np.exp(-aa[:,None]*lam)*lam**3
        rc += (widths/2)**4/24*(remainder@nc)
        rd += (widths/2)**4/24*(remainder@nd)
        step_values=(-np.expm1(-aa[:,None]*lam)/lam)[:,:,None]*F[None,:,:]
        denominators=np.linalg.svd(step_values,compute_uv=False)[:,-1]
        for a,b,dt,rci,rdi,lower in zip(aa,bb,widths,rc,rd,denominators):
            attenuation=np.exp(-self.alpha*dt)
            integral=-np.expm1(-self.alpha*dt)/self.alpha
            en=min(attenuation*ec+integral*rci,
                   np.sqrt(attenuation*ec*ec+integral*rdi*rdi))
            worst=max(worst,max(ec,en)/lower)
            ec=en; intervals+=1
            if early_reject and worst > target:
                return dict(passed=False,steady_bound=float(steady),step_bound=float(worst/(1-worst)) if worst<1 else None,
                            reason='time enclosure failed',intervals=intervals,failure_time=float(b),seconds=time.perf_counter()-start)
        tail=la.norm(rss,2)+sum(nd/lam*np.exp(-lam*tail_time))
        worst=max(worst,max(ec,tail/np.sqrt(self.alpha))/den(tail_time))
        step=float(worst/(1-worst)) if worst<1 else None
        return dict(passed=bool(steady<=epsilon and worst<=target),steady_bound=float(steady),
                    step_bound=step,intervals=intervals,seconds=time.perf_counter()-start,
                    failure_time=float(tail_time) if worst>target else None,
                    reason='all-time enclosures evaluated')
