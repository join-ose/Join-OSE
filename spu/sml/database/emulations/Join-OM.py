import jax.numpy as jnp
import jax
import numpy as np
import sml.utils.emulation as emulation
import spu.intrinsic as si


def emul_Join_UN(mode: emulation.Mode.MULTIPROCESS):
    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode
        )
        emulator.up()    

        def GenBitPerm(k,n):
            f0=1-k
            f1=k
            s0=jax.lax.associative_scan(jnp.add, f0)
            f1=f1.at[0].set(f1[0]+s0[n-1])
            s1=jax.lax.associative_scan(jnp.add, f1)
            t=k*(s1-s0)
            return s0+t-1  
        
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
            # e2 = e2.at[0].set(1)
            e2 = jnp.concatenate([jnp.ones(1, dtype=jnp.int32),e2[1:]])
            num1 = jnp.arange(1, nxy+1,dtype=jnp.int32)
            num2 = jnp.arange(nxy, 0, -1, dtype=jnp.int32)


            x1=e1*num1+(1-e1)*(nxy)  
            e_flip=1-e1
            perm1=GenBitPerm(e_flip,nxy)
            y1=jnp.array(si.invperm(x1,perm1))
            y1_roll=jnp.roll(y1,1)
            # y1_roll=y1_roll.at[0].set(0)
            y1_roll = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),y1_roll[1:]])
            y1 = y1-y1_roll
            y1=si.perm(y1,perm1)
            y1 = jax.lax.associative_scan(jnp.add, y1)
            y1 = jnp.roll(y1,1)
            # y1 = y1.at[0].set(0)
            y1 = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32),y1[1:]])
            y1 = y1+num1-1
            
            x2=e2*num2+(1-e2)*(nxy)  
            perm2=GenBitPerm(e2,nxy)
            y2=jnp.array(si.invperm(x2,perm2))
            y2_roll=jnp.roll(y2,-1)
            # y2_roll=y2_roll.at[nxy-1].set(0)
            y2_roll = jnp.concatenate([y2_roll[:-1],jnp.zeros(1, dtype=jnp.int32)])
            y2 = y2-y2_roll
            y2=si.perm(y2,perm2)
            y2 = jax.lax.associative_scan(jnp.add, y2)
            y2 = y2+num1-1

            inv_sigma2 = jnp.concatenate([y1, y2],axis=0)
            sigma1 = jnp.concatenate([sigma1, sigma1+nxy],axis=0)
            sigma3 = si.invperm(sigma1,inv_sigma2)

            return sigma3  

        def getsortpermforxyx(skx,sky,nx,ny):
            sigma = getsortperm(skx,sky,nx,ny)
            zeros = jnp.zeros(2*nx+ny,dtype=jnp.int32)
            ones=jnp.ones(ny,dtype=jnp.int32)
            e = jnp.concatenate([zeros,ones], axis=0)
            e = si.perm(e,sigma)
            rho = GenBitPerm(e,nx*2+ny*2)
            sigma = si.invperm(sigma,rho)
            return sigma[:2*nx+ny]
            



        def AHK(skx,sky,spx,spy,nx,ny):
            skxy=jnp.concatenate((skx,sky,skx))
            sigma=jnp.argsort(skxy)
            spxy = jnp.concatenate((spx,spy,-spx))
            sorted_spxy = si.perm(spxy,sigma)
            sorted_spxy=jax.lax.associative_scan(jnp.add, sorted_spxy)
            spxy = si.invperm(sorted_spxy,sigma)
            return spxy[nx:nx+ny+1]
        
        def InnerJoin(skx,sky,spx,spy,nx,ny):
            sigma=getsortpermforxyx(skx,sky,nx,ny)
            spxy = jnp.concatenate((spx,spy,-spx))
            sorted_spxy = si.perm(spxy,sigma)
            sorted_spxy=jax.lax.associative_scan(jnp.add, sorted_spxy)
            spxy = si.invperm(sorted_spxy,sigma)
            return spxy[nx:nx+ny+1]



        nx=2**20
        ny=2**20

        n=(nx+ny)//4
        k=np.random.randint(0, 2**30,size=n, dtype=np.uint32)
        kx=np.random.randint(0, 2**30,size=nx-n, dtype=np.uint32)
        ky=np.random.randint(0, 2**30,size=ny-n, dtype=np.uint32)
        kx=jnp.concatenate([k, kx], axis=0)
        ky=jnp.concatenate([k, ky], axis=0)
        keyx = jax.random.PRNGKey(0)
        keyy = jax.random.PRNGKey(1)
        kx=jax.random.permutation(keyx,kx)
        ky=jax.random.permutation(keyy,ky)
        px=np.random.randint(0, 2**30,size=nx, dtype=np.uint32)
        py=np.random.randint(0, 2**30,size=ny, dtype=np.uint32)

        skx,sky,spx,spy=emulator.seal(kx,ky,px,py)

        r1=emulator.run(AHK,static_argnums=(4,5,))(skx,sky,spx,spy,nx,ny)
        r2=emulator.run(InnerJoin,static_argnums=(4,5,))(skx,sky,spx,spy,nx,ny)
        print(jnp.array_equal(r1, r2))


    finally:
        emulator.down()


if __name__ == "__main__":
    emul_Join_UN(emulation.Mode.MULTIPROCESS)
