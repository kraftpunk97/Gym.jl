using Test
using Gym

speclist_ = collect(speclist())

@testset "Basic tests" begin
    @testset "$envname" for envname in speclist_

        # Create an environment using no_render mode
        env1 = make(envname, :no_render)
        env2 = make(envname, :no_render)
        @test true

        #Get its action space
        actionspace = env1.action_space

        #Set its seed
        seed!(env1, 42)
        seed!(env2, 42)
        @test true

        #Reset it
        obs11 = reset!(env1)
        obs21 = reset!(env2)
        @test all(obs11 .== obs21)

        #Get an outcome
        seed!(actionspace, 42)
        action = sample(actionspace)
        obs12 = step!(env1, action)
        obs22 = step!(env2, action)
        @test all(obs12 .== obs22)

        #Set another seed
        seed!(env1, 42)
        seed!(env2, 42)

        obs13 = reset!(env1)
        obs23 = reset!(env2)

        @test all(obs11 .== obs13)
        @test all(obs21 .== obs23)
    end
end
