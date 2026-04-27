import jax.numpy as jnp
import jax
import numpy as np
import sml.utils.emulation as emulation
import spu.intrinsic as si


def emul_OSE(mode: emulation.Mode.MULTIPROCESS):
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
        
        def sort_test(skx,nx):
            skxy=jnp.concatenate([skx,skx], axis=0)
            sigma=jnp.argsort(skxy)
            # skxy1=jnp.concatenate([skx,sky], axis=0)
            # sigma1=jnp.argsort(skxy1)
            return sigma
        
        def  sort_test2(skx,nx):
            sigma1=jnp.argsort(skx)
            skxy=jnp.array(si.perm(skx,sigma1))
            skxy1=jnp.roll(skxy,-1)
            f=jnp.equal(skxy,skxy1)
            e1=1-f
            e1=e1.at[nx-1].set(1)
            e2 = jnp.roll(e1,1)
            # e2 = e2.at[0].set(1)
            e2 = jnp.concatenate([jnp.ones(1, dtype=jnp.int32),e2[1:]])
            num1 = jnp.arange(1, nx+1,dtype=jnp.int32)
            num2 = jnp.arange(nx, 0, -1, dtype=jnp.int32)


            x1=e1*num1+(1-e1)*(nx)  
            e_flip=1-e1
            perm1=GenBitPerm(e_flip,nx)
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
            
            x2=e2*num2+(1-e2)*(nx)  
            perm2=GenBitPerm(e2,nx)
            y2=jnp.array(si.invperm(x2,perm2))
            y2_roll=jnp.roll(y2,-1)
            # y2_roll=y2_roll.at[nxy-1].set(0)
            y2_roll = jnp.concatenate([y2_roll[:-1],jnp.zeros(1, dtype=jnp.int32)])
            y2 = y2-y2_roll
            y2=si.perm(y2,perm2)
            y2 = jax.lax.associative_scan(jnp.add, y2)
            y2 = y2+num1-1

            inv_sigma2 = jnp.concatenate([y1, y2],axis=0)
            sigma1 = jnp.concatenate([sigma1, sigma1+nx],axis=0)
            sigma3 = si.invperm(sigma1,inv_sigma2)

            return sigma3  

        

        nx=2**8
        kx=np.random.randint(0, 2**30,size=nx, dtype=np.uint32)
        keyx = jax.random.PRNGKey(1)
        kx=jax.random.permutation(keyx,kx)
        skx=emulator.seal(kx)

        # r1=emulator.run(sort_test,static_argnums=(1,))(skx,nx)
        r2=emulator.run(sort_test2,static_argnums=(1,))(skx,nx)
        # print(jnp.array_equal(r1, r2))


    finally:
        emulator.down()


if __name__ == "__main__":
    emul_OSE(emulation.Mode.MULTIPROCESS)
