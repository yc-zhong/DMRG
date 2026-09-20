using LinearAlgebra
using FileIO
using ITensors
using ITensorMPS
using TensorKit

# Small, reproducible same-maxdim comparison point for the migration harness.
# Run from /Users/kani/Workplace:
#   DMRG_BENCH_D=16 DMRG_BENCH_SWEEPS=4 \
#     julia +1.11.5 --project=DMRG/converter DMRG/test/hubbard_obc_sameD_itensor.jl

include("hubbard_obc_ed_reference.jl")
using .HubbardOBCEDReference

function benchmark_mpo(sites; Lx, Ly, t, ty, U, left_edge_h)
    os = OpSum()
    for bond in nearest_neighbor_bonds_reference(Lx, Ly; t=t, ty=ty)
        i, j, a = bond.i, bond.j, bond.amplitude
        os += -a, "Cdagup", i, "Cup", j
        os += -a, "Cdagup", j, "Cup", i
        os += -a, "Cdagdn", i, "Cdn", j
        os += -a, "Cdagdn", j, "Cdn", i
    end
    for x in 1:Lx, y in 1:Ly
        i = site_index_reference(x, y, Lx, Ly)
        os += U, "Nupdn", i
        field = left_edge_h * left_edge_profile_reference(x, y, Lx, Ly)
        if !iszero(field)
            os += field, "Nup", i
            os += -field, "Ndn", i
        end
    end
    return MPO(os, sites)
end

function neel_labels(Lx, Ly)
    labels = fill("Emp", Lx * Ly)
    for x in 1:Lx, y in 1:Ly
        labels[site_index_reference(x, y, Lx, Ly)] = iseven(x + y) ? "Up" : "Dn"
    end
    return labels
end

Lx, Ly = 3, 2
Nup, Ndn = 3, 3
t, ty = 1.0, 1.0
U = 12.0
left_edge_h = 0.1
D = parse(Int, get(ENV, "DMRG_BENCH_D", "16"))
nsweeps = parse(Int, get(ENV, "DMRG_BENCH_SWEEPS", "4"))

BLAS.set_num_threads(1)
sites = siteinds("Electron", Lx * Ly; conserve_qns=true)
H = benchmark_mpo(sites; Lx, Ly, t, ty, U, left_edge_h)
psi0 = MPS(sites, neel_labels(Lx, Ly))
observer = DMRGObserver()

elapsed = @elapsed energy, psi = dmrg(
    H,
    psi0;
    nsweeps=nsweeps,
    maxdim=D,
    cutoff=0.0,
    observer=observer,
    outputlevel=0,
    eigsolve_krylovdim=5,
    eigsolve_tol=1e-4,
    eigsolve_maxiter=1,
)

H_exact, _ = dense_fixed_sector_hamiltonian(
    Lx,
    Ly,
    Nup,
    Ndn;
    t,
    ty,
    U,
    left_edge_h,
)
exact_energy = eigmin(Hermitian(H_exact))

println("engine=ITensorMPS")
println("geometry=$(Lx)x$(Ly)")
println("D=$D")
println("nsweeps=$nsweeps")
println("energies=", energies(observer))
println("max_truncation_errors=", truncerrors(observer))
println("final_energy=", energy)
println("exact_energy=", exact_energy)
println("energy_error=", energy - exact_energy)
println("maxlinkdim=", maxlinkdim(psi))
println("elapsed_seconds=", elapsed)

function dense_tensorkit_state(directory, nsites)
    A = convert(Array, FileIO.load(joinpath(directory, "1.jld2"))["ttn_tem"])
    size(A, 1) == 1 || error("Expected a one-dimensional left boundary")
    state = reshape(A, size(A, 2), size(A, 3))
    for site in 2:nsites
        A = convert(Array, FileIO.load(joinpath(directory, "$(site).jld2"))["ttn_tem"])
        size(state, 2) == size(A, 1) || error("Broken virtual bond before site $site")
        next = zeros(promote_type(eltype(state), eltype(A)), size(state, 1), 4, size(A, 3))
        for physical in 1:4
            next[:, physical, :] = state * A[:, physical, :]
        end
        state = reshape(next, size(state, 1) * 4, size(A, 3))
    end
    size(state, 2) == 1 || error("Expected a one-dimensional right boundary")
    vector = vec(state)
    return vector / norm(vector)
end

function diagonal_observables(vector, nsites)
    local_n = (0.0, 1.0, 1.0, 2.0)
    local_sz = (0.0, 0.5, -0.5, 0.0)
    local_d = (0.0, 0.0, 0.0, 1.0)
    probabilities = abs2.(vector)
    densities = zeros(nsites)
    spins = zeros(nsites)
    doublons = zeros(nsites)
    nn = zeros(nsites, nsites)
    ss = zeros(nsites, nsites)
    for linear in eachindex(probabilities)
        index = linear - 1
        weight = probabilities[linear]
        values_n = zeros(nsites)
        values_sz = zeros(nsites)
        for site in 1:nsites
            physical = mod(div(index, 4^(site - 1)), 4) + 1
            values_n[site] = local_n[physical]
            values_sz[site] = local_sz[physical]
            densities[site] += weight * values_n[site]
            spins[site] += weight * values_sz[site]
            doublons[site] += weight * local_d[physical]
        end
        nn .+= weight .* (values_n * values_n')
        ss .+= weight .* (values_sz * values_sz')
    end
    return densities, spins, doublons, nn, ss
end

if haskey(ENV, "TENSORKIT_MPS_DIR")
    vector = dense_tensorkit_state(ENV["TENSORKIT_MPS_DIR"], Lx * Ly)
    itensor_dense = vec(Array(reduce(*, psi), sites...))
    itensor_dense ./= norm(itensor_dense)
    tk_n, tk_sz, tk_d, tk_nn, tk_ss = diagonal_observables(vector, Lx * Ly)
    it_n = real.(expect(psi, "Ntot"))
    it_sz = real.(expect(psi, "Sz"))
    it_d = real.(expect(psi, "Nupdn"))
    it_nn = real.(correlation_matrix(psi, "Ntot", "Ntot"))
    it_ss = real.(correlation_matrix(psi, "Sz", "Sz"))
    pairs = ((1, 2), (1, 4), (2, 5), (3, 6))
    tk_charge_connected = [tk_nn[i, j] - tk_n[i] * tk_n[j] for (i, j) in pairs]
    it_charge_connected = [it_nn[i, j] - it_n[i] * it_n[j] for (i, j) in pairs]
    tk_spin_connected = [tk_ss[i, j] - tk_sz[i] * tk_sz[j] for (i, j) in pairs]
    it_spin_connected = [it_ss[i, j] - it_sz[i] * it_sz[j] for (i, j) in pairs]

    println("tensorkit_norm=", real(dot(vector, vector)))
    println("normalized_overlap=", abs(dot(itensor_dense, vector)))
    println("tensorkit_total_N=", sum(tk_n))
    println("itensor_total_N=", sum(it_n))
    println("tensorkit_total_Sz=", sum(tk_sz))
    println("itensor_total_Sz=", sum(it_sz))
    println("max_local_density_difference=", maximum(abs.(tk_n - it_n)))
    println("max_local_Sz_difference=", maximum(abs.(tk_sz - it_sz)))
    println("max_local_double_occupancy_difference=", maximum(abs.(tk_d - it_d)))
    println("correlator_pairs=", collect(pairs))
    println("tensorkit_charge_connected=", tk_charge_connected)
    println("itensor_charge_connected=", it_charge_connected)
    println("max_charge_connected_difference=",
            maximum(abs.(tk_charge_connected - it_charge_connected)))
    println("tensorkit_spin_connected=", tk_spin_connected)
    println("itensor_spin_connected=", it_spin_connected)
    println("max_spin_connected_difference=",
            maximum(abs.(tk_spin_connected - it_spin_connected)))
end
