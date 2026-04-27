import jax.numpy as jnp
import jax
import numpy as np
import sml.utils.emulation as emulation
import spu.intrinsic as si


def emul_Join_NN(mode: emulation.Mode.MULTIPROCESS):

    def GenBitPerm(k,n):
        f0=1-k
        f1=k
        s0=jax.lax.associative_scan(jnp.add, f0)
        f1=f1.at[0].set(f1[0]+s0[n-1])
        s1=jax.lax.associative_scan(jnp.add, f1)
        t=k*(s1-s0)
        return s0+t-1  
    
    def  GetPostion(s_replicate,n,m):
        num = jnp.arange(0, m+n,dtype=jnp.int32)
        i = jnp.arange(n,dtype=jnp.int32)
        num1 = jnp.arange(0, m,dtype=jnp.int32)
        prefix_sum = jax.lax.associative_scan(jnp.add, s_replicate)
        g = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), prefix_sum[:-1]])
        h = g+i
        rho = jnp.array(si.buildPerm(h,num))
        zeros = jnp.zeros(m,dtype=jnp.int32)
        ones=jnp.ones(n,dtype=jnp.int32)
        e0 = jnp.concatenate([ones,zeros,ones+1,zeros], axis=0)
        e1 = jnp.array(si.invperm(e0,rho))

        e_2 = jnp.concatenate([jnp.ones(1, dtype=jnp.int32), e1[1+n+m:]])
        e_2 = jax.lax.associative_scan(jnp.add, e_2)
        inv_perm = GenBitPerm(e1[:n+m],n+m)
        e_2 = jnp.array(si.invperm(e_2,inv_perm))[:m]+num1
        d = g + i * 2
        f = prefix_sum + (i + 1) * 2-1
        ret = jnp.concatenate([d, f,e_2],axis=0)
        return ret
    
    def getsortperm(skx,sky,nx,ny):
        nxy = nx+ny
        skxy=jnp.concatenate([skx,sky], axis=0)
        sigma1=jnp.argsort(skxy)
        skxy=jnp.array(si.perm(skxy,sigma1))
        skxy1=jnp.roll(skxy,-1)
        f=jnp.equal(skxy,skxy1)
        e1=1-f
        e1=e1.at[nxy-1].set(1)
        e2 = jnp.roll(e1,1)
        e2 = jnp.concatenate([jnp.ones(1, dtype=jnp.int32),e2[1:]])
        num1 = jnp.arange(1, nxy+1,dtype=jnp.int32)
        num2 = jnp.arange(nxy, 0, -1, dtype=jnp.int32)


        x1=e1*num1+(1-e1)*(nxy)  
        e_flip=1-e1
        perm1=GenBitPerm(e_flip,nxy)
        y1=jnp.array(si.invperm(x1,perm1))
        y1_roll=jnp.roll(y1,1)
        y1_roll = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),y1_roll[1:]])
        y1 = y1-y1_roll
        y1=si.perm(y1,perm1)
        y1 = jax.lax.associative_scan(jnp.add, y1)
        y1 = jnp.roll(y1,1)
        y1 = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),y1[1:]])
        y1 = y1+num1-1
        
        x2=e2*num2+(1-e2)*(nxy)  
        perm2=GenBitPerm(e2,nxy)
        y2=jnp.array(si.invperm(x2,perm2))
        y2_roll=jnp.roll(y2,-1)
        y2_roll = jnp.concatenate([y2_roll[:-1],jnp.zeros(1, dtype=jnp.int32)])
        y2 = y2-y2_roll
        y2=si.perm(y2,perm2)
        y2 = jax.lax.associative_scan(jnp.add, y2)
        y2 = y2+num1-1

        inv_sigma2 = jnp.concatenate([y1, y2],axis=0)
        sigma1 = jnp.concatenate([sigma1, sigma1+nxy],axis=0)
        sigma3 = si.invperm(sigma1,inv_sigma2)

        return  sigma3


    def InnerJoin1(skx,sky,spx,spy,x_zeros_0,y_zeros_0,nx,ny,m):
        N = (nx+ny)*2
        x_zeros = jnp.zeros(nx,dtype=jnp.int32)
        x_ones=jnp.ones(nx,dtype=jnp.int32)
        x_neg_ones = jnp.full(nx, -1,dtype=jnp.int32)
        y_zeros = jnp.zeros(ny,dtype=jnp.int32)
        y_ones=jnp.ones(ny,dtype=jnp.int32)
        y_neg_ones = jnp.full(ny, -1,dtype=jnp.int32)
        e_0 = jnp.concatenate([x_zeros, y_zeros, x_ones, y_ones], axis=0)
        e_1 = jnp.concatenate([x_zeros, y_ones, x_zeros, y_ones], axis=0)
        replicate_x = jnp.concatenate([x_zeros, y_ones, x_zeros, y_neg_ones], axis=0)
        replicate_y = jnp.concatenate([x_ones, y_zeros, x_neg_ones, y_zeros], axis=0)
        sigma = getsortperm(skx,sky,nx,ny)
    
        set = jnp.concatenate([replicate_x,replicate_y,e_0,e_1])
        set=si.perm(set,sigma)
        s_replicate_xy = set[:2*N]
        s_e_0 = set[2*N:3*N]
        s_e_1 = set[3*N:]
        s_replicate_xy = jax.lax.associative_scan(jnp.add, s_replicate_xy)
        s_replicate_xy = si.invperm(s_replicate_xy,sigma)
        s_replicate_x=jnp.array(s_replicate_xy[:N])
        s_replicate_y=jnp.array(s_replicate_xy[N:])
   
        inv_perm1 = GenBitPerm(s_e_1,N)
  
        s_e_0 = si.invperm(s_e_0,inv_perm1)
        inv_perm2 = GenBitPerm(s_e_0,(nx+ny)*2)
        inv_perm1 = si.invperm(sigma,inv_perm1)
        perm = si.invperm(inv_perm1,inv_perm2)

        perm1 = perm[:nx]
        perm2 = perm[nx:nx+ny]-nx
        

        set_perm1 = jnp.concatenate([skx,spx,s_replicate_x[nx+ny:2*nx+ny]])
        set_perm2 = jnp.concatenate([sky,spy,s_replicate_x[nx:nx+ny],s_replicate_y[nx:nx+ny]])
        set_perm1 = si.perm(set_perm1,perm1)
        set_perm2 = si.perm(set_perm2,perm2)
        skx = jnp.array(set_perm1[:nx])
        spx = jnp.array(set_perm1[nx:2*nx])
        s_replicate_x = set_perm1[2*nx:]
        sky = jnp.array(set_perm2[:ny])
        spy = jnp.array(set_perm2[ny:2*ny])
        sR = jnp.array(set_perm2[2*ny:3*ny])
        s_replicate_y = jnp.array(set_perm2[3*ny:])

        rhoX = GetPostion(s_replicate_x,nx,m)
        rhoY = GetPostion(s_replicate_y,ny,m)
        s_replicate_x= jnp.array(s_replicate_x)
 


        x_zeros_1 = jnp.zeros(m,dtype=jnp.int32)
        x_ones_1=jnp.ones(2*nx,dtype=jnp.int32)
        y_zeros_1 = jnp.zeros(m,dtype=jnp.int32)
        y_ones_1=jnp.ones(2*ny,dtype=jnp.int32)
        y_zeros_2 = jnp.zeros(ny,dtype=jnp.int32)
        y_ones_2 = jnp.ones(m,dtype=jnp.int32)
        
        e_0 = jnp.concatenate([x_ones_1,x_zeros_1], axis=0)
        e_1 = jnp.concatenate([y_ones_1,y_zeros_1], axis=0)
        skx = jnp.concatenate([skx,-skx,x_zeros_0], axis=0)
        spx = jnp.concatenate([spx,-spx,x_zeros_0], axis=0)
        spy = jnp.concatenate([spy,-spy,y_zeros_0], axis=0)
        a = jnp.concatenate([y_zeros_2,-s_replicate_y,y_ones_2], axis=0)
        s_replicate_x = jnp.concatenate([s_replicate_x,-s_replicate_x,x_zeros_1], axis=0)
        s_replicate_y = jnp.concatenate([s_replicate_y,-s_replicate_y,y_zeros_1], axis=0)
        sR = jnp.concatenate([sR,-sR,y_zeros_1], axis=0)

        
        dx_set = jnp.concatenate([skx,spx,s_replicate_x,e_0])
        dy_set = jnp.concatenate([spy,a,s_replicate_y,sR,e_1])
        dx_set = si.invperm(dx_set,rhoX)
        dy_set = si.invperm(dy_set,rhoY)
        Mx = m+2*nx
        My = m+2*ny
        skx = dx_set[:Mx]
        spx = dx_set[Mx:2*Mx]
        s_replicate_x = dx_set[2*Mx:3*Mx]
        s_e_0 = dx_set[3*Mx:]
        spy = dy_set[:My]
        a = dy_set[My:2*My]
        s_replicate_y = dy_set[2*My:3*My]
        sR = dy_set[3*My:4*My]
        s_e_1 = dy_set[4*My:]

        skx = jax.lax.associative_scan(jnp.add, skx)
        spx = jax.lax.associative_scan(jnp.add, spx)
        spy = jax.lax.associative_scan(jnp.add, spy)
        a = jax.lax.associative_scan(jnp.add, a)
        s_replicate_x = jax.lax.associative_scan(jnp.add, s_replicate_x)
        s_replicate_y = jax.lax.associative_scan(jnp.add, s_replicate_y)
        sR = jax.lax.associative_scan(jnp.add, sR)

        
        inv_perm_x = GenBitPerm(s_e_0,m+2*nx)
        inv_perm_y = GenBitPerm(s_e_1,m+2*ny)
        
        inv_perm_x_set = jnp.concatenate([skx,spx,s_replicate_x])
        inv_perm_y_set = jnp.concatenate([spy,s_replicate_y,sR,a])
        inv_perm_x_set = si.invperm(inv_perm_x_set,inv_perm_x)
        inv_perm_y_set = si.invperm(inv_perm_y_set,inv_perm_y)
        skx = inv_perm_x_set[:Mx]
        spx = inv_perm_x_set[Mx:2*Mx]
        s_replicate_x = inv_perm_x_set[2*Mx:]
        spy = inv_perm_y_set[:My]
        s_replicate_y = inv_perm_y_set[My:2*My]
        sR = inv_perm_y_set[2*My:3*My]
        a = inv_perm_y_set[3*My:]


        
        skx = skx[:m]
        spx = spx[:m]
        spy = spy[:m]
        sR = sR[:m]-y_ones_2
        s_replicate_x = s_replicate_x[:m]-y_ones_2
        s_replicate_y = s_replicate_y[:m]-y_ones_2
        a = a[:m]-y_ones_2
        i = jnp.arange(1, m+1,dtype=jnp.int32)
        b = i - sR*s_replicate_y + a*s_replicate_x-1
        spy = si.invperm(spy,b)

        return skx,spx,spy


    

    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode
        )
        emulator.up()  
        kx = np.array([1, 4, 5, 2, 2], dtype=np.uint32)     
        ky = np.array([1, 2, 2, 1, 2], dtype=np.uint32)  
        px = np.array([11, 14, 15, 12, 13], dtype=np.uint32) 
        py = np.array([21, 23, 24, 22, 25], dtype=np.uint32) 
        nx = 5
        ny  =5
        ux, cx = jnp.unique(kx, return_counts=True)
        uy, cy = jnp.unique(ky, return_counts=True)
        count_x = dict(zip(np.array(ux), np.array(cx)))
        count_y = dict(zip(np.array(uy), np.array(cy)))
        m = sum(count_x[k] * count_y[k] for k in count_x if k in count_y)
        x_zeros_0 = np.zeros(m, dtype=np.uint32)
        y_zeros_0 = np.zeros(m, dtype=np.uint32)
        
       
        skx,sky,spx,spy=emulator.seal(kx,ky,px,py)

        kx,ky,py=emulator.run(InnerJoin1,static_argnums=(6,7,8,))(skx,sky,spx,spy,x_zeros_0,y_zeros_0,nx,ny,m)

        print('kx: ',kx)
        print('ky: ',ky)
        print('py: ',py)


    finally:
        emulator.down()


if __name__ == "__main__":
    emul_Join_NN(emulation.Mode.MULTIPROCESS)
