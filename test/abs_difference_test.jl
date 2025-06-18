using QuantusJL
using Test

@testset "absDif_test" begin
    a = [3.5, 3.0, 3.2]
    b = [2.5, 2.0, 2.2]
    c= QuantusJL.abs_difference(a, b)

    @test c==1
    @test isa(c, Float64) 
end
