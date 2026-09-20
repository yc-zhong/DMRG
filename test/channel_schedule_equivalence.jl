#!/usr/bin/env julia

using Test
import MPI

MPI.Initialized() || MPI.Init()
const comm=MPI.COMM_WORLD
const root=0
const rank=Int(MPI.Comm_rank(comm))
const sizeofmpi=Int(MPI.Comm_size(comm))

haskey(ENV,"ENVIRONMENT_TMP_ROOT") ||
    error("Set ENVIRONMENT_TMP_ROOT to a dedicated empty test directory")
include(joinpath(@__DIR__,"..","model","Square_Hubbard_U1_U1_OBC_LeftEdge.jl"))
include(joinpath(@__DIR__,"..","MPS.jl"))

function legacy_update_plans(position)
    left=Vector{Vector{Int64}}()
    right=Vector{Vector{Int64}}()
    for term in terms_keys
        if position==term[1]
            push!(left,[term[1],term[3]])
        elseif term[1]<position && term[2]>position
            push!(left,[term[1],term[3]])
        elseif term[2]==position
            push!(left,copy(term))
        end
        if term[1]==position
            push!(right,copy(term))
        elseif term[1]<position && term[2]>position
            push!(right,[term[2],term[3]])
        elseif term[2]==position
            push!(right,[term[2],term[3]])
        end
    end
    unique!(left); sort!(left;by=Tuple)
    unique!(right); sort!(right;by=Tuple)
    return left,right
end

function legacy_haction_plan(position)
    touching=filter(term->position in term[1:2] || position+1 in term[1:2],terms_keys)
    crossing=filter(term->term[1]<position && term[2]>position+1,terms_keys)
    grouped=Dict{Vector{Int64},Vector{Int64}}()
    for term in touching
        if term[1]<position
            key=[0,term[2],term[3]]
            haskey(grouped,key) ? push!(grouped[key],term[1]) : (grouped[key]=copy(term))
        elseif term[2]>position+1
            key=[-1,term[1],term[3]]
            haskey(grouped,key) ? push!(grouped[key],term[2]) : (grouped[key]=copy(term))
        else
            grouped[copy(term)]=copy(term)
        end
    end
    for term in crossing
        key=[term[1],term[3]]
        haskey(grouped,key) ? push!(grouped[key],term[2]) : (grouped[key]=copy(term))
    end
    result=collect(values(grouped))
    sort!(result;by=Tuple)
    append!(result,[[0],[-1]])
    return result
end

L_num=parameter["L"][1]*parameter["L"][2]
@testset "canonical exact channel schedule" begin
    for position in 1:L_num
        left,right=legacy_update_plans(position)
        @test _channel_schedule.left_update[position]==left
        @test _channel_schedule.right_update[position]==right
        @test _channel_schedule.close_left[position]==
            filter(term->term[2]==position,terms_keys)
        @test _channel_schedule.close_right[position]==
            filter(term->term[1]==position,terms_keys)
    end
    for position in 1:L_num-1
        @test _channel_schedule.haction[position]==legacy_haction_plan(position)
    end
    if parameter["L"]==[32,6]
        stats=channel_schedule_stats()
        @test stats.raw_terms==692
        @test stats.raw_crossing_states==2552
        @test stats.compact_prefix_states==2242
    end
end

rank==root && println("channel_schedule_equivalence=PASS stats=",channel_schedule_stats())
MPI.Finalized() || MPI.Finalize()
