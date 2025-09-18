
"""
    int_xcir(a::Int)

Recursively, compute the integral of x^a sqrt(1-x^2) from 0 to 1
"""
function int_xcir(a::Int)
    if Bool(a & 1)
        res = 1 / 3
    else
        res = π / 4
    end
    while a > 1
        res *= 1 - 3 / (a + 2)
        a -= 2
    end
    return res
end

"""
    int_xasin(a::Int)

Compute the integral of x^a asin(x)
"""
function int_xasin(a::Int)
    return 0.5 * π / (a + 1) - a / (a + 1) * int_xcir(a - 1)
end


"""
    asin_approx_coeff(n::Int)

Compute the best coefficients for polynomial approximation of asin(x)
"""
function asin_approx_coeff(n::Int)
    A = Matrix{Float64}(undef, n, n)
    for l ∈ 1:n
        for k ∈ 1:n
            A[k, l] = 1 / (2l + 2k + 3) - 1 / (2k + 3) - 1 / (2l + 3) + 1 / 3
        end
    end
    b = Vector{Float64}(undef, n)
    for k ∈ 1:n
        b[k] = 1 / 12 + 1 / (2k + 2) - 1 / (2k + 3) - (2k + 1) / (k + 1) * int_xcir(2k) / π
    end
    return A \ b
end



"""
    asin_approx(coef::Vector, x)

Compute the polynomial approximation of arcsin, given the coefficients
"""
function asin_approx(coef::Vector, x)
    result = similar(x)
    copy!(result, x)
    for (k, c) ∈ enumerate(coef)
        @. result += c * (x^(2k + 1) - x)
    end
    return result
end
