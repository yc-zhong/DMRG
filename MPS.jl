using TensorKit
using LinearAlgebra
using TensorOperations
using ProgressMeter
import MPI
using Random

function Terms_keys()
    if rank==0
        TT=collect(keys(terms))
    end
    TT=MPI.bcast(rank==0 ? TT : nothing,root,comm)
    return TT
end

const terms_keys=sort!(Terms_keys();by=Tuple)

# Compile the implicit hopping prefix/suffix DAG once. The legacy code rebuilt
# these exact lists from every raw Hamiltonian term at every site (and, for the
# H-action, at every Krylov application). No coefficient or QN block is
# truncated here.
function _compile_channel_schedule(term_table,L_num::Int)
    left_update=[Vector{Vector{Int64}}() for _ in 1:L_num]
    right_update=[Vector{Vector{Int64}}() for _ in 1:L_num]
    left_delete=[Vector{Vector{Int64}}() for _ in 1:L_num]
    right_delete=[Vector{Vector{Int64}}() for _ in 1:L_num]
    close_left=[Vector{Vector{Int64}}() for _ in 1:L_num]
    close_right=[Vector{Vector{Int64}}() for _ in 1:L_num]
    haction=[Vector{Vector{Int64}}() for _ in 1:max(L_num-1,0)]

    for position in 1:L_num
        for term in term_table
            if position==term[1]
                push!(left_update[position],[term[1],term[3]])
            elseif term[1]<position<term[2]
                push!(left_update[position],[term[1],term[3]])
            elseif term[2]==position
                push!(left_update[position],copy(term))
            end

            if term[1]==position
                push!(right_update[position],copy(term))
            elseif term[1]<position<term[2]
                push!(right_update[position],[term[2],term[3]])
            elseif term[2]==position
                push!(right_update[position],[term[2],term[3]])
            end

            if position==term[1] || term[1]<position<term[2]
                push!(left_delete[position],[term[1],term[3]])
            end
            if term[1]<position<term[2] || term[2]==position
                push!(right_delete[position],[term[2],term[3]])
            end
            term[2]==position && push!(close_left[position],copy(term))
            term[1]==position && push!(close_right[position],copy(term))
        end
        unique!(left_update[position]); sort!(left_update[position];by=Tuple)
        unique!(right_update[position]); sort!(right_update[position];by=Tuple)
        unique!(left_delete[position]); sort!(left_delete[position];by=Tuple)
        unique!(right_delete[position]); sort!(right_delete[position];by=Tuple)
        sort!(close_left[position];by=Tuple)
        sort!(close_right[position];by=Tuple)
    end

    for position in 1:L_num-1
        grouped=Dict{Tuple{Vararg{Int64}},Vector{Int64}}()
        for term in term_table
            touches=(term[1]==position || term[2]==position ||
                term[1]==position+1 || term[2]==position+1)
            crosses=term[1]<position && term[2]>position+1
            (touches || crosses) || continue
            key=if touches && term[1]<position
                (0,term[2],term[3])
            elseif touches && term[2]>position+1
                (-1,term[1],term[3])
            elseif crosses
                (term[1],term[3])
            else
                Tuple(term)
            end
            if haskey(grouped,key)
                endpoint=touches && term[1]<position ? term[1] : term[2]
                push!(grouped[key],endpoint)
            else
                grouped[key]=copy(term)
            end
        end
        groups=collect(values(grouped))
        sort!(groups;by=Tuple)
        append!(groups,[[0],[-1]])
        if @isdefined cal_excited
            cal_excited==true && push!(groups,[-2])
        end
        haction[position]=groups
    end
    return (left_update=left_update,right_update=right_update,
        left_delete=left_delete,right_delete=right_delete,
        close_left=close_left,close_right=close_right,haction=haction)
end

const _channel_schedule=_compile_channel_schedule(
    terms_keys,parameter["L"][1]*parameter["L"][2])

function channel_schedule_stats()
    raw_crossing=0
    compact_crossing=0
    L_num=parameter["L"][1]*parameter["L"][2]
    for cut in 1:L_num-1
        raw_crossing+=count(term->term[1]<=cut<term[2],terms_keys)
        compact_crossing+=count(entry->length(entry)==2,
            _channel_schedule.left_update[cut])
    end
    return (raw_terms=length(terms_keys),raw_crossing_states=raw_crossing,
        compact_prefix_states=compact_crossing,
        haction_groups=sum(length,_channel_schedule.haction),
        haction_groups_max=maximum(length,_channel_schedule.haction))
end

const _window_transient_store=Dict{String,Tuple{Any,Int}}()
const _window_transient_bytes=Ref(0)
const _window_transient_peak_bytes=Ref(0)
const _window_transient_hits=Ref(0)
const _window_transient_backing_fallbacks=Ref(0)

function _stable_term_owners_enabled()
    return _parse_bool("ENV_STABLE_TERM_OWNERS",
        get(ENV,"ENV_STABLE_TERM_OWNERS","false"))
end

function _deterministic_owner(values::Vararg{Int})
    state=UInt64(0xcbf29ce484222325)
    for value in values
        state=(state ⊻ reinterpret(UInt64,Int64(value)))*UInt64(0x100000001b3)
    end
    return Int(mod(state,UInt64(sizeofmpi)))
end

function _mini_environment_owner(direction::Symbol,endpoint::Int,operator::Int,
        legacy_index::Int)
    _stable_term_owners_enabled() || return legacy_index%sizeofmpi
    direction_code=direction==:left ? 1 : direction==:right ? 2 :
        error("Invalid mini-environment direction: $direction")
    return _deterministic_owner(direction_code,endpoint,operator)
end

function _full_term_owner(direction::Symbol,term,legacy_index::Int;consumer::Bool=false)
    _stable_term_owners_enabled() ||
        return (legacy_index+(consumer ? 2 : 0))%sizeofmpi
    endpoint=direction==:left ? term[1] : direction==:right ? term[2] :
        error("Invalid full-term direction: $direction")
    return _mini_environment_owner(direction,endpoint,term[3],legacy_index)
end

function _window_transient_limit_bytes()
    budget_gb=parse(Float64,get(ENV,"ENV_WINDOW_TRANSIENT_GB","0"))
    isfinite(budget_gb) && budget_gb>=0 ||
        error("ENV_WINDOW_TRANSIENT_GB must be finite and nonnegative")
    return floor(Int,budget_gb*1024^3)
end

function _window_transient_enabled()
    enabled=_window_transient_limit_bytes()>0
    enabled && !_stable_term_owners_enabled() && error(
        "ENV_WINDOW_TRANSIENT_GB requires ENV_STABLE_TERM_OWNERS=true")
    return enabled
end

function _window_transient_delete!(fname::AbstractString)
    key=_environment_key(fname)
    entry=pop!(_window_transient_store,key,nothing)
    entry===nothing || (_window_transient_bytes[]-=entry[2])
    return entry!==nothing
end

function _window_transient_save!(value,fname::AbstractString)
    key=_environment_key(fname)
    _window_transient_delete!(key)
    bytes=Base.summarysize(value)
    limit=_window_transient_limit_bytes()
    if bytes<=limit && _window_transient_bytes[]+bytes<=limit
        _remove_environment_files_without_count!(key)
        _window_transient_store[key]=(value,bytes)
        _window_transient_bytes[]+=bytes
        _window_transient_peak_bytes[]=max(
            _window_transient_peak_bytes[],_window_transient_bytes[])
        return :resident
    end
    tensor_save(value,key)
    _window_transient_backing_fallbacks[]+=1
    return :backing
end

function _window_transient_take!(fname::AbstractString)
    key=_environment_key(fname)
    entry=get(_window_transient_store,key,nothing)
    if entry!==nothing
        _window_transient_hits[]+=1
        return entry[1]
    end
    value=tensor_load(key)
    return value
end

function _window_transient_publish!(fname::AbstractString)
    key=_environment_key(fname)
    entry=get(_window_transient_store,key,nothing)
    entry===nothing && return :backing
    tensor_save(entry[1],key)
    return :resident_publish
end

function _window_transient_snapshot!(source::AbstractString,destination::AbstractString)
    key=_environment_key(source)
    entry=get(_window_transient_store,key,nothing)
    if entry!==nothing
        tensor_save(entry[1],destination)
        return (mode=:resident_publish,bytes=entry[2],path=destination)
    end
    return environment_clone_backing!(source,destination)
end

function _window_transient_import!(fname::AbstractString)
    key=_environment_key(fname)
    value=tensor_load(key)
    bytes=Base.summarysize(value)
    limit=_window_transient_limit_bytes()
    if bytes<=limit && _window_transient_bytes[]+bytes<=limit
        _window_transient_store[key]=(value,bytes)
        _window_transient_bytes[]+=bytes
        _window_transient_peak_bytes[]=max(
            _window_transient_peak_bytes[],_window_transient_bytes[])
        return :resident
    end
    return :backing
end

function _window_transient_finish!(direction::Symbol)
    local_error=nothing
    for key in keys(_window_transient_store)
        try
            _environment_existing_file(key)
        catch err
            local_error="$(key): $(sprint(showerror,err))"
            break
        end
    end
    failed_ranks=MPI.Allreduce(Int(local_error!==nothing),+,comm)
    if failed_ranks>0
        local_error===nothing || println(stderr,
            "WINDOW_TRANSIENT_BACKING_ERROR rank=$rank direction=$direction $local_error")
        MPI.Barrier(comm)
        error("Transient $direction backing validation failed on $failed_ranks ranks")
    end
    entries=length(_window_transient_store)
    bytes=_window_transient_bytes[]
    peak=_window_transient_peak_bytes[]
    hits=_window_transient_hits[]
    fallbacks=_window_transient_backing_fallbacks[]
    empty!(_window_transient_store)
    _window_transient_bytes[]=0
    total_entries=MPI.Allreduce(entries,+,comm)
    total_bytes=MPI.Allreduce(bytes,+,comm)
    aggregate_peak=MPI.Allreduce(peak,+,comm)
    total_hits=MPI.Allreduce(hits,+,comm)
    total_fallbacks=MPI.Allreduce(fallbacks,+,comm)
    rank==root && println("WINDOW_TRANSIENT_STATS direction=",direction,
        " released_entries=",total_entries,
        " released_bytes=",total_bytes,
        " aggregate_peak_bytes=",aggregate_peak,
        " hits=",total_hits,
        " backing_fallbacks=",total_fallbacks)
    _window_transient_peak_bytes[]=0
    _window_transient_hits[]=0
    _window_transient_backing_fallbacks[]=0
    return nothing
end

function _window_transient_reset!()
    isempty(_window_transient_store) || error(
        "Window transient store is not empty: $(length(_window_transient_store)) entries")
    _window_transient_bytes[]=0
    _window_transient_peak_bytes[]=0
    _window_transient_hits[]=0
    _window_transient_backing_fallbacks[]=0
    return nothing
end

function _window_env_take(fname::AbstractString,transient::Bool)
    return transient ? _window_transient_take!(fname) : tensor_load(fname)
end

function _window_env_save(value,fname::AbstractString,transient::Bool)
    transient ? _window_transient_save!(value,fname) : tensor_save(value,fname)
    return nothing
end

function _window_env_delete(fname::AbstractString,transient::Bool)
    if transient
        _window_transient_delete!(fname) || write_zero(fname)
    else
        write_zero(fname)
    end
    return nothing
end


function left_block_add_mini(before::TensorMap,position::Int64,ttn::TensorMap,terms::Vector{TensorMap};boson::Int64=1)
    if position==1
        @tensor A[-1,-2,-3]:=ttn[2,1,-1]*terms[1][1,3,-2]*conj(ttn[2,3,-3])
        return A
    elseif position==2
        if boson>0
            @tensor A[-1,-2,-3]:=before[1,-2,2]*ttn[1,3,-1]*conj(ttn[2,3,-3])
        else
            F=TensorKit_matrix("F")
            if @isdefined Space_type
                F=TensorKit_matrix("F",string(dim(space(ttn)[2])))
            end
            @tensor A[-1,-2,-3]:=before[2,-2,3]*ttn[2,1,-1]*F[1,4]*conj(ttn[3,4,-3])
        end
        return A
    elseif position==0
        @tensor A[-1,-2]:=before[1,2,4]*ttn[1,3,-1]*terms[2][2,3,5]*conj(ttn[4,5,-2])
        return A
    end
end


function right_block_add_mini(before::TensorMap,position::Int64,ttn::TensorMap,terms::Vector{TensorMap};boson::Int64=1)
    if position==1
        @tensor A[-1,-2,-3]:=ttn[-1,1,2]*terms[2][-2,1,3]*conj(ttn[-3,3,2])
        return A
    elseif position==2
        if boson>0
            @tensor A[-1,-2,-3]:=before[1,-2,2]*ttn[-1,3,1]*conj(ttn[-3,3,2])
        else
            F=TensorKit_matrix("F")
            if @isdefined Space_type
                F=TensorKit_matrix("F",string(dim(space(ttn)[2])))
            end
            @tensor A[-1,-2,-3]:=before[2,-2,3]*ttn[-1,1,2]*F[1,4]*conj(ttn[-3,4,3])
        end
        return A
    elseif position==0
        @tensor A[-1,-2]:=before[1,2,4]*ttn[-1,3,1]*terms[1][3,5,2]*conj(ttn[-2,5,4])
        return A
    end
end


function initial_blocks_mini(save_file::String;p::Int64=1)
    if rank==0
        p_bar = Progress(parameter["L"][1]*parameter["L"][2],desc=out_put_file*"_block",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
    end
    if p==1
        for i in parameter["L"][1]*parameter["L"][2]:-1:1
            if rank==0
                ProgressMeter.next!(p_bar)
            end
            ttn=tensor_load(save_file*"/"*string(i))
            update_blocks_right_mini(ttn,i)
            MPI.Barrier(comm)
        end  
    else
        for i in 1:parameter["L"][1]*parameter["L"][2]
            if rank==0
                ProgressMeter.next!(p_bar)
            end
            ttn=tensor_load(save_file*"/"*string(i))
            update_blocks_left_mini(ttn,i)
            MPI.Barrier(comm)
        end 
    end
    return 0
end


function update_blocks_left_mini(ttn::TensorMap,position::Int64;transient::Bool=false)
    terms_e=_channel_schedule.left_update[position]

    for ii in 1:length(terms_e)
        i=terms_e[ii]
        owner=length(i)==3 ? _full_term_owner(:left,i,ii) :
            _mini_environment_owner(:left,i[1],i[end],ii)
        if owner==rank
            if length(i)==3
                before=_window_env_take(tmp_file*"/"*string(hash([i[1],[i[end],1,position-1]])),transient)
                _window_env_save(left_block_add_mini(before,0,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([[i[1],i[2]],[i[end],1,0]])),transient)
            elseif i[1]==position 
                _window_env_save(left_block_add_mini(ttn,1,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
            elseif i[1]<position && length(i)==2
                before=_window_env_take(tmp_file*"/"*string(hash([i[1],[i[end],1,position-1]])),transient)
                _window_env_save(left_block_add_mini(before,2,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
            end
        end
    end
    before = nothing
    return 0
end

function update_blocks_right_mini(ttn::TensorMap,position::Int64;transient::Bool=false)
    terms_e=_channel_schedule.right_update[position]

    for ii in 1:length(terms_e)
        i=terms_e[ii]
        owner=length(i)==3 ? _full_term_owner(:right,i,ii) :
            _mini_environment_owner(:right,i[1],i[end],ii)
        if owner==rank
            if length(i)==3
                before=_window_env_take(tmp_file*"/"*string(hash([i[2],[i[end],1,position+1]])),transient)
                _window_env_save(right_block_add_mini(before,0,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([[i[1],i[2]],[i[end],1,0]])),transient)
            elseif i[1]==position
                _window_env_save(right_block_add_mini(ttn,1,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
            elseif i[1]>position && length(i)==2
                before=_window_env_take(tmp_file*"/"*string(hash([i[1],[i[end],1,position+1]])),transient)
                _window_env_save(right_block_add_mini(before,2,ttn,Ham_matrix[i[end]];boson=i[end]),tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
            end
        end
    end
    before = nothing
    return 0
end

function delete_blocks_right_mini(position::Int64;transient::Bool=false)
    L_num=parameter["L"][1]*parameter["L"][2]
    1<=position<=L_num || return 0
    terms_e=_channel_schedule.right_delete[position]
    for ii in 1:length(terms_e)
        i=terms_e[ii]
        if _mini_environment_owner(:right,i[1],i[end],ii)==rank
            _window_env_delete(tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
        end
    end
    return 0
end

function delete_blocks_left_mini(position::Int64;transient::Bool=false)
    L_num=parameter["L"][1]*parameter["L"][2]
    1<=position<=L_num || return 0
    terms_e=_channel_schedule.left_delete[position]

    for ii in 1:length(terms_e)
        i=terms_e[ii]
        if _mini_environment_owner(:left,i[1],i[end],ii)==rank
            _window_env_delete(tmp_file*"/"*string(hash([i[1],[i[end],1,position]])),transient)
        end
    end
    return 0
end

function left_block_add(before::TensorMap,position::Int64,ttn::TensorMap,sites;transient::Bool=false)
    if 0%sizeofmpi==rank
        @tensor A[-1,-2]:=before[1,4]*ttn[1,2,-1]*conj(ttn[4,2,-2])
    end
    if 1%sizeofmpi==rank
        if @isdefined A
            @tensor A[-1,-2]:=A[-1,-2]+ttn[2,1,-1]*terms_onsite[position][1,3]*conj(ttn[2,3,-2])
        else
            @tensor A[-1,-2]:=ttn[2,1,-1]*terms_onsite[position][1,3]*conj(ttn[2,3,-2])
        end
    end

    for ii in 1:length(sites)
        i=sites[ii]
        if _full_term_owner(:left,i,ii;consumer=true)==rank
            term_block=_window_env_take(tmp_file*"/"*string(hash([[i[1],i[2]],[i[3],1,0]])),transient)
            if @isdefined A
                A=A+term_block*parameter[terms[i]]
            else
                A=term_block*parameter[terms[i]]
            end
        end
    end
    if !(@isdefined A)
        @tensor A[-1,-2]:=TensorMap(zeros,space(ttn,3),space(ttn,3))[-1,-2]
    end
    mpi_reduce_tensormap_sum!(A,comm;root=root)
    return A
end

function right_block_add(before::TensorMap,position::Int64,ttn::TensorMap,sites;transient::Bool=false)
    if 0%sizeofmpi==rank
        @tensor A[-1,-2]:=before[1,4]*ttn[-1,2,1]*conj(ttn[-2,2,4])
    end
    if 1%sizeofmpi==rank
        if @isdefined A
            @tensor A[-1,-2]:=A[-1,-2]+ttn[-1,1,2]*terms_onsite[position][1,3]*conj(ttn[-2,3,2])
        else
            @tensor A[-1,-2]:=ttn[-1,1,2]*terms_onsite[position][1,3]*conj(ttn[-2,3,2])
        end
    end
    for ii in 1:length(sites)
        i=sites[ii]
        if _full_term_owner(:right,i,ii;consumer=true)==rank
            term_block=_window_env_take(tmp_file*"/"*string(hash([[i[1],i[2]],[i[3],1,0]])),transient)
            if @isdefined A
                A=A+term_block*parameter[terms[i]]
            else
                A=term_block*parameter[terms[i]]
            end
        end
    end
    if !(@isdefined A)
        @tensor A[-1,-2]:=TensorMap(zeros,space(ttn,1),space(ttn,1))[-1,-2]
    end
    mpi_reduce_tensormap_sum!(A,comm;root=root)
    return A
end


function find_terms(position::Int64;direction=0)
    if direction==0
        return _channel_schedule.close_left[position]
    elseif direction==1
        return _channel_schedule.close_right[position]
    end
    error("direction must be 0 (close-left) or 1 (close-right)")
end

function initial_blocks(save_file::String;p::Int64=1)
    L=parameter["L"]
    if p!=1
        ttn=tensor_load(save_file*"/"*string(1))
        @tensor A[-1,-2]:=ttn[2,1,-1]*terms_onsite[1][1,3]*conj(ttn[2,3,-2])
        if rank==0
            tensor_save(A,tmp_file*"/"*string(hash([0,1])))
        end
        for i in 2:L[1]*L[2]  
            sites=find_terms(i)
            ttn=tensor_load(save_file*"/"*string(i))
            A=left_block_add(A,i,ttn,terms,sites)
            delete_full_term_blocks!(sites;direction=:left)
            if rank==0
                tensor_save(A,tmp_file*"/"*string(hash([0,i])))
            end
        end
    else
        ttn=tensor_load(save_file*"/"*string(L[1]*L[2]))
        @tensor A[-1,-2]:=ttn[-1,1,2]*terms_onsite[L[1]*L[2]][1,3]*conj(ttn[-2,3,2])
        if rank==1%sizeofmpi
            @tensor A2[-1,-2]:=ttn[-1,1,2]*conj(ttn[-2,1,2])
            tensor_save(A2,tmp_file*"/"*string(hash([2,L[1]*L[2]])))
        end
        if rank==0
            tensor_save(A,tmp_file*"/"*string(hash([1,L[1]*L[2]])))
        end
        for i in L[1]*L[2]-1:-1:1
            sites=find_terms(i,direction=1)
            ttn=tensor_load(save_file*"/"*string(i))
            A=right_block_add(A,i,ttn,sites)
            delete_full_term_blocks!(sites;direction=:right)
            if rank==1%sizeofmpi
                @tensor A2[-1,-2]:=A2[1,2]*ttn[-1,3,1]*conj(ttn[-2,3,2])
                tensor_save(A2,tmp_file*"/"*string(hash([2,i])))
            end
            if rank==0
                tensor_save(A,tmp_file*"/"*string(hash([1,i])))
            end
        end
    end
    return 0
end

function pre_add!(tensor::TensorMap,value::Float64,tensor_file::String)
    # `A` is a read-only contraction operand. `tensor` is the separately
    # allocated accumulator and is the only object modified here.
    A=haction_environment_load_readonly(tensor_file)
    @tensor tensor[-1,-2,-3] += A[-1,-2,-3]*value
    A=nothing
    nothing
end

function pre_comuting_left!(position::Int64,i::Vector{Int64},sites::Vector{Int64},before::TensorMap)
    for j in sites
        pre_add!(before,parameter[terms[[j,i[2],i[3]]]],tmp_file*"/"*string(hash([j,[i[3],1,position-1]])))
    end
    nothing
end

function pre_comuting_right!(position::Int64,i::Vector{Int64},sites::Vector{Int64},after::TensorMap)
    for j in sites
        pre_add!(after,parameter[terms[[i[1],j,i[3]]]],tmp_file*"/"*string(hash([j,[i[3],1,position+2]])))
    end
    nothing
end

function enviroment_one_mini(ttn::TensorMap,E::TensorMap,position::Int64,i::Vector{Int64})
    term1=Ham_matrix[i[3]][1]
    term2=Ham_matrix[i[3]][2]
    LT=0.0
    #value=parameter[terms[[i[1],i[2],i[3]]]]
    if i[2]==position
        # Scalar multiplication materializes the mutable accumulator; the
        # solve-scoped pinned source remains read-only.
        before=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[1],[i[3],1,position-1]])))*parameter[terms[[i[1],i[2],i[3]]]]
        if length(i)>=4
            pre_comuting_left!(position,i,i[4:end],before)
        end
        LT=time()
        @tensor E[-1,-2,-3,-4] += before[1,2,-1]*term2[2,3,-2]*ttn[1,3,-3,-4]
    elseif i[1]==position+1
        after=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[2],[i[3],1,position+2]])))*parameter[terms[[i[1],i[2],i[3]]]]
        if length(i)>=4
            pre_comuting_right!(position,i,i[4:end],after)
        end
        LT=time()
        @tensor E[-1,-2,-3,-4] += ttn[-1,-2,3,1]*term1[3,-3,2]*after[1,2,-4]
    elseif i[1]==position && i[2]>position+1
        after=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[2],[i[3],1,position+2]])))*parameter[terms[[i[1],i[2],i[3]]]]
        if length(i)>=4
            pre_comuting_right!(position,i,i[4:end],after)
        end
        LT=time()
        if i[3]>0
            @tensor E[-1,-2,-3,-4] += ttn[-1,3,-3,1]*term1[3,-2,2]*after[1,2,-4]
        else
            F=TensorKit_matrix("F")
            if @isdefined Space_type
                F=TensorKit_matrix("F",string(dim(space(ttn)[3])))
            end
            @tensor E[-1,-2,-3,-4] += ttn[-1,4,1,2]*term1[4,-2,3]*F[1,-3]*after[2,3,-4]
        end
    elseif i[2]==position+1 && i[1]<position
        before=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[1],[i[3],1,position-1]])))*parameter[terms[[i[1],i[2],i[3]]]]
        if length(i)>=4
            pre_comuting_left!(position,i,i[4:end],before)
        end
        LT=time()
        if i[3]>0
            @tensor E[-1,-2,-3,-4] += before[1,2,-1]*term2[2,4,-3]*ttn[1,-2,4,-4]
        else
            F=TensorKit_matrix("F")
            if @isdefined Space_type
                F=TensorKit_matrix("F",string(dim(space(ttn)[2])))
            end
            @tensor E[-1,-2,-3,-4] += before[1,3,-1]*term2[3,4,-3]*ttn[1,2,4,-4]*F[2,-2]
        end
    elseif i[1]==position && i[2]==position+1
        value=parameter[terms[[i[1],i[2],i[3]]]]
        LT=time()
        @tensor E[-1,-2,-3,-4] += term1[4,-2,1]*term2[1,2,-3]*ttn[-1,4,2,-4]*value
    elseif i[2]>position+1 && i[1]<position
        before=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[1],[i[3],1,position-1]])))
        after=haction_environment_load_readonly(tmp_file*"/"*string(hash([i[2],[i[3],1,position+2]])))*parameter[terms[[i[1],i[2],i[3]]]]
        if length(i)>=4
            pre_comuting_right!(position,i,i[4:end],after)
        end
        LT=time()
        if i[3]>0
            @tensor E[-1,-2,-3,-4] += before[1,4,-1]*ttn[1,-2,-3,2]*after[2,4,-4]
        else
            F1=TensorKit_matrix("F")
            F2=TensorKit_matrix("F")
            if @isdefined Space_type
                F1=TensorKit_matrix("F",string(dim(space(ttn)[2])))
                F2=TensorKit_matrix("F",string(dim(space(ttn)[3])))
            end
            @tensor E[-1,-2,-3,-4] += before[3,6,-1]*ttn[3,1,2,4]*F1[1,-2]*F2[2,-3]*after[4,6,-4]
        end
    end
    before=nothing
    after=nothing
    return LT
end

function enviroment_one_mini(ttn::TensorMap,E::TensorMap,position::Int64,i::Int64)
    LT=0.0
    L=parameter["L"]
    if i[1]==-1
        LT=time()
        @tensor E[-1,-2,-3,-4] += ttn[-1,1,-3,-4]*terms_onsite[position][1,-2]
        @tensor E[-1,-2,-3,-4] += ttn[-1,-2,2,-4]*terms_onsite[position+1][2,-3]
    elseif i[1]==0
        if position-1>0 && position+1<L[1]*L[2]
            blocks_left =haction_environment_load_readonly(tmp_file*"/"*string(hash([1,position-1])))
            blocks_right =haction_environment_load_readonly(tmp_file*"/"*string(hash([1,position+2])))
            LT=time()
            @tensor E[-1,-2,-3,-4] += blocks_left[1,-1]*ttn[1,-2,-3,-4]
            @tensor E[-1,-2,-3,-4] += ttn[-1,-2,-3,2]*blocks_right[2,-4]
        elseif position==1
            blocks_right =haction_environment_load_readonly(tmp_file*"/"*string(hash([1,position+2])))
            LT=time()
            @tensor E[-1,-2,-3,-4] += ttn[-1,-2,-3,1]*blocks_right[1,-4]
        elseif position==L[1]*L[2]-1
            blocks_left =haction_environment_load_readonly(tmp_file*"/"*string(hash([1,position-1])))
            LT=time()
            @tensor E[-1,-2,-3,-4] += blocks_left[1,-1]*ttn[1,-2,-3,-4]
        end
    elseif i[1]==-2
        base_ttn1=tensor_load(base_file*"/"*string(position))
        base_ttn2=tensor_load(base_file*"/"*string(position+1))
        if position-1>0 && position+1<L[1]*L[2]
            blocks_left =haction_environment_load_readonly(tmp_file*"/"*string(hash([3,position-1])))
            blocks_right =haction_environment_load_readonly(tmp_file*"/"*string(hash([4,position+2])))
            LT=time()
            @tensor blocks_left[-1,-2,-3,-4]:=blocks_left[1,-1]*base_ttn1[1,-2,3]*base_ttn2[3,-3,2]*blocks_right[2,-4]
            @tensor E[-1,-2,-3,-4] += -blocks_left[1,2,3,4]*conj(ttn[1,2,3,4])*blocks_left[-1,-2,-3,-4]*base_energy
        elseif position==1
            blocks_right =haction_environment_load_readonly(tmp_file*"/"*string(hash([4,position+2])))
            LT=time()
            @tensor blocks_left[-1,-2,-3,-4]:=base_ttn1[-1,-2,3]*base_ttn2[3,-3,2]*blocks_right[2,-4]
            @tensor E[-1,-2,-3,-4] += -blocks_left[1,2,3,4]*conj(ttn[1,2,3,4])*blocks_left[-1,-2,-3,-4]*base_energy
        elseif position==L[1]*L[2]-1
            blocks_left =haction_environment_load_readonly(tmp_file*"/"*string(hash([3,position-1])))
            LT=time()
            @tensor blocks_left[-1,-2,-3,-4]:=blocks_left[1,-1]*base_ttn1[1,-2,3]*base_ttn2[3,-3,-4]
            @tensor E[-1,-2,-3,-4] += -blocks_left[1,2,3,4]*conj(ttn[1,2,3,4])*blocks_left[-1,-2,-3,-4]*base_energy
        end
    end
    blocks_left=nothing
    blocks_right=nothing
    return LT
end

function enviroment_one(ttn::TensorMap,E::TensorMap,position::Int64,i::Vector{Int64})
    ST=time()
    LT=0.0
    if length(i)>=3
        LT=enviroment_one_mini(ttn,E,position,i)
    elseif length(i)==1
        LT=enviroment_one_mini(ttn,E,position,i[1])
    end
    return time()-LT
end

function terms_E_reconfig(position::Int64)
    1<=position<=length(_channel_schedule.haction) ||
        error("Invalid two-site Hamiltonian position: $position")
    return _channel_schedule.haction[position]
end

function enviroment_general(ttn::TensorMap,position::Int64;MAX_NUM=40)
    ST=time()
    terms_E=terms_E_reconfig(position)
    terms_E=MPI.bcast(rank==0 ? terms_E : nothing,root,comm)
    @tensor E[-1,-2,-3,-4]:=TensorMap(zeros,Float64,codomain(ttn),domain(ttn))[-1,-2,-3,-4]
    com_time=0.0
    N=length(terms_E)
    scheduler=lowercase(strip(get(ENV,"H_ACTION_SCHEDULER","dynamic_workers")))
    scheduler in ("dynamic_workers","static_all_ranks") ||
        error("H_ACTION_SCHEDULER must be dynamic_workers or static_all_ranks")
    if sizeofmpi==1 || scheduler=="static_all_ranks"
        local_compute=0.0
        for i in 1:N
            if (i-1)%sizeofmpi==rank
                p_time=enviroment_one(ttn,E,position,terms_E[i])
                local_compute+=p_time
            end
        end
        com_time=MPI.Allreduce(local_compute,+,comm)
    else
        nworkers = sizeofmpi - 1
        send_mesg = Array{Int64}(undef, 1)
        recv_mesg = Array{Int64}(undef, 1)
        time_computing = Array{Float64}(undef, 1)
        if rank==0
            idx_recv = 0
            idx_sent = 1
            status_workers = ones(nworkers).*-1
            send_buffers = [zeros(Int64,1) for _ in 1:nworkers]
            for dst in 1:nworkers
                if idx_sent > N
                    break
                end
                send_buffers[dst][1] = idx_sent
                MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
                idx_sent += 1
                status_workers[dst] = 1
            end
            while idx_recv != N
                for dst in 1:nworkers
                    if status_workers[dst] == 1
                        ismessage = MPI.Iprobe(comm; source=dst, tag=dst+MAX_NUM)
                        if ismessage
                            MPI.Recv!(time_computing, comm; source=dst, tag=dst+MAX_NUM)
                            idx_recv += 1
                            com_time+=time_computing[1]
                            if idx_sent <= N
                                send_buffers[dst][1] = idx_sent
                                MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
                                idx_sent += 1
                                status_workers[dst] = 1
                            else
                                status_workers[dst] = -1
                            end
                        end
                    end
                end
            end
            for dst in 1:nworkers
                # Termination message to worker
                send_buffers[dst][1] = -1
                MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
            end
        else
            while true
                ismessage = MPI.Iprobe(comm; source=root, tag=rank+MAX_NUM)
                if ismessage
                    # Receives message
                    MPI.Recv!(recv_mesg, comm; source=root, tag=rank+MAX_NUM)
                    # Termination message from root
                    if recv_mesg[1] == -1
                        break
                    end
                    # Apply function to array
                    p_time=enviroment_one(ttn,E,position,terms_E[recv_mesg[1]])
                    time_computing[1]=p_time
                    MPI.Send(time_computing, comm; dest=root, tag=rank+MAX_NUM)
                end
            end
        end
    end
    MPI.Barrier(comm)
    mpi_reduce_tensormap_sum!(E,comm;root=root)
    #println(com_time/(time()-ST))
    worker_count=scheduler=="static_all_ranks" ? sizeofmpi : max(sizeofmpi-1,1)
    return E,com_time/(time()-ST)/worker_count,length(terms_E)
end

function matrix_product(vector::TensorMap,position::Int64)
    started=time_ns()
    result=enviroment_general(vector,position)
    add_segment_timing!(:hamiltonian_action,(time_ns()-started)/1.0e9)
    return result
end


function _eigsh_lanczos_impl(ttn1::TensorMap,ttn2::TensorMap,position::Int64,
        num_krylov_vecs::Int64=parse(Int,get(ENV,"KRYLOV_DIM","3")),
        numeig::Int64=1,tol::Float64=parse(Float64,get(ENV,"KRYLOV_TOL","1e-14")),
        delta::Float64=1E-6,ndiag::Int64=20,reorthogonalize::Bool=false)
    #vector_n= ncon([ttn[position],ttn[position+1]],[[-1,-2,2],[2,-3,-4]],[false,false];output=[-1,-2,-3])
    @tensor vector_n[-1,-2,-3,-4] :=ttn1[-1,-2,2]*ttn2[2,-3,-4]
    norms_vector_n = Float64[]
    diag_elements = Float64[]
    krylov_vecs = TensorMap[]
    eigvalsold = Float64[]
    PA_NUM=1
    re_time=0
    process=true
    MPI.Barrier(comm)
    for it in 1:num_krylov_vecs
        if rank==0
            norm_vector_n = norm(vector_n)
            if abs(norm_vector_n) < delta
                process=false
            end
            if process
                norms_vector_n=vcat(norms_vector_n,[norm_vector_n])
                vector_n = vector_n / norms_vector_n[end]
                # store the Lanczos vector for later
                if reorthogonalize
                    for v in krylov_vecs
                        norm_sq =dot(vector_n,v)
                        vector_n -= norm_sq * v
                    end
                end
                krylov_vecs=vcat(krylov_vecs,[vector_n])
                tensor_save(vector_n,joinpath(tmp_file,"enviroment"))
            end
        end
        process=MPI.bcast(rank==0 ? process : nothing,root,comm)
        if process==false
            break
        end
        # Rank 0 overwrites the same Krylov-vector key every iteration.  All
        # ranks must invalidate the previous read-through-cache entry only
        # after that write is globally visible.  Keep immutable Hamiltonian
        # environments cached across Krylov applications.
        MPI.Barrier(comm)
        environment_cache_invalidate!(joinpath(tmp_file,"enviroment"))
        MPI.Barrier(comm)
        vector_n=tensor_load(joinpath(tmp_file,"enviroment"))
        A_vector_n,re_time,PA_NUM = matrix_product(vector_n,position)
        if rank==0
            diag_elements=vcat(diag_elements,[dot(A_vector_n,vector_n)])
            if it > 1
                # diagonalize the effective Hamiltonian
                A_tridiag=diagm(diag_elements)
                A_tridiag[diagind(A_tridiag, 1)]= norms_vector_n[2:end]
                A_tridiag[diagind(A_tridiag, -1)]= norms_vector_n[2:end]
                B=SymTridiagonal((A_tridiag+adjoint(A_tridiag))/2)
                eigvalues,u=eigen(Array(B))
                #println(eigvalues)
                #println(eigvalsold[1]-eigvalues[1])
                if (abs(eigvalsold[1]-eigvalues[1])< tol) && (it>=3)
                    process=false
                end
                eigvalsold = eigvalues
            else
                eigvalsold = diag_elements
            end
            if process
                if it > 1
                    A_vector_n = A_vector_n - (krylov_vecs[end] * diag_elements[end])
                    A_vector_n = A_vector_n - (krylov_vecs[end-1] * norms_vector_n[end])
                else
                    A_vector_n = A_vector_n - (krylov_vecs[end] * diag_elements[end])
                end
                vector_n = A_vector_n
            end
        end
        process=MPI.bcast(rank==0 ? process : nothing,root,comm)
        if process==false
            break
        end
    end
    if rank==0
        A_tridiag=diagm(diag_elements)
        A_tridiag[diagind(A_tridiag, 1)]= norms_vector_n[2:end]
        A_tridiag[diagind(A_tridiag, -1)]= norms_vector_n[2:end]
        B=SymTridiagonal((A_tridiag+adjoint(A_tridiag))/2)
        eigvalues,u=eigen(Array(B))
        eigenvectors = TensorMap[]
        for n2 in 1:min(numeig, length(eigvalues))
            state = TensorMap(zeros,Float64,codomain(vector_n),domain(vector_n))
            for n1 in 1:length(krylov_vecs)
                state += krylov_vecs[n1] * (u[n1, n2])
                #println(u[n1, n2])
            end
            eigenvectors=vcat(eigenvectors,[state / norm(state)])
        end
        Ham_save(eigenvectors,joinpath(tmp_file,"eigenvectors"))
    end
    MPI.Barrier(comm)
    eigenvectors=tensor_load(joinpath(tmp_file,"eigenvectors"))
    eigvalues=MPI.bcast(rank==0 ? eigvalues : nothing,root,comm)
    return real(eigvalues[1:numeig]), eigenvectors,re_time,PA_NUM
end

"""
Run one two-site Lanczos solve with a bounded, rank-local pin set for immutable
Hamiltonian environments. The mutable Krylov vector and result scratch keys
are forbidden explicitly, and the pin set is released in `finally` on both a
normal bond boundary and an exception.
"""
function eigsh_lanczos(ttn1::TensorMap,ttn2::TensorMap,position::Int64,
        num_krylov_vecs::Int64=parse(Int,get(ENV,"KRYLOV_DIM","3")),
        numeig::Int64=1,tol::Float64=parse(Float64,get(ENV,"KRYLOV_TOL","1e-14")),
        delta::Float64=1E-6,ndiag::Int64=20,reorthogonalize::Bool=false)
    forbidden=(joinpath(tmp_file,"enviroment"),
        joinpath(tmp_file,"eigenvectors"))
    return with_haction_pin_cache(position;forbidden_keys=forbidden) do
        _eigsh_lanczos_impl(ttn1,ttn2,position,num_krylov_vecs,numeig,tol,
            delta,ndiag,reorthogonalize)
    end
end

function write_data_txt(file_name::String,data::String)
    open(file_name*".txt", "a+") do file
        write(file, data*"\n")
    end
    return 0
end

function initial_ttn(ttn::Vector{TensorMap})
    L_num=parameter["L"][1]*parameter["L"][2]
    for i in 1:L_num-1
        ttn[i],ttn[i+1]=expand_dim(ttn[i],ttn[i+1],100)
        ttn[i],ttn[i+1],trunc_err=compress_dim(ttn[i],ttn[i+1],100)
    end
    ttn=norm_TN(1,ttn,L=parameter["L"])
    return ttn
end

function update_mini(rev::Bool,i::Int64,ttn1::TensorMap,ttn2::TensorMap)
    L=parameter["L"]
    L_num=L[1]*L[2]
    if rev
        update_blocks_right_mini(ttn2,i+1)
        MPI.Barrier(comm)
        if i==1
            update_blocks_right_mini(ttn1,i)
        end
    else
        update_blocks_left_mini(ttn1,i)
        MPI.Barrier(comm)
        if i==L_num-1
            update_blocks_left_mini(ttn2,i+1)
        end
    end
    return 0
end

function updata_super(rev::Bool,i::Int64,ttn1::TensorMap,ttn2::TensorMap)
    L=parameter["L"]
    L_num=L[1]*L[2]
    if rev
        if i==L_num-1
            @tensor A[-1,-2]:=ttn2[-1,1,2]*terms_onsite[L_num][1,3]*conj(ttn2[-2,3,2])
            if rank==0
                tensor_save(A,tmp_file*"/"*string(hash([1,L_num])))
            end
            if 1%sizeofmpi==rank && @isdefined cal_excited
                if cal_excited==true
                    base_ttn=tensor_load(base_file*"/"*string(L_num))
                    @tensor A[-1,-2]:=base_ttn[-1,1,2]*conj(ttn2[-2,1,2])
                    tensor_save(A,tmp_file*"/"*string(hash([4,L_num])))
                end
            end
        else
            A=tensor_load(tmp_file*"/"*string(hash([1,i+2])))
            sites=find_terms(i+1,direction=1)
            A=right_block_add(A,i+1,ttn2,sites)
            delete_full_term_blocks!(sites;direction=:right)
            if rank==0
                tensor_save(A,tmp_file*"/"*string(hash([1,i+1])))
            end

            if 1%sizeofmpi==rank && @isdefined cal_excited
                if cal_excited==true
                    base_ttn=tensor_load(base_file*"/"*string(i+1))
                    A=tensor_load(tmp_file*"/"*string(hash([4,i+2])))
                    @tensor A[-1,-2]:=A[1,2]*base_ttn[-1,3,1]*conj(ttn2[-2,3,2])
                    tensor_save(A,tmp_file*"/"*string(hash([4,i+1])))
                end
            end
        end
    else
        if i >1
            A=tensor_load(tmp_file*"/"*string(hash([1,i-1])))
            sites=find_terms(i)
            A=left_block_add(A,i,ttn1,sites)
            delete_full_term_blocks!(sites;direction=:left)
            if rank==0
                tensor_save(A,tmp_file*"/"*string(hash([1,i])))
            end
            if 1%sizeofmpi==rank && @isdefined cal_excited
                if cal_excited==true
                    base_ttn=tensor_load(base_file*"/"*string(i))
                    A=tensor_load(tmp_file*"/"*string(hash([2,i-1])))
                    @tensor A[-1,-2]:=A[1,2]*base_ttn[1,3,-1]*conj(ttn1[2,3,-2])
                    tensor_save(A,tmp_file*"/"*string(hash([3,i])))
                end
            end
        else
            if rank==0
                @tensor A[-1,-2]:=ttn1[2,1,-1]*terms_onsite[1][1,3]*conj(ttn1[2,3,-2])
                tensor_save(A,tmp_file*"/"*string(hash([1,1])))
            end
            if 1%sizeofmpi==rank && @isdefined cal_excited
                if cal_excited==true
                    base_ttn=tensor_load(base_file*"/"*string(1))
                    @tensor A[-1,-2]:=base_ttn[2,1,-1]*conj(ttn1[2,1,-2])
                    tensor_save(A,tmp_file*"/"*string(hash([3,1])))
                end
            end
        end
    end
    return 0
end

const _active_environment_window_sites=Ref{Union{Nothing,Int}}(nothing)
const _active_environment_window_dimension=Ref{Union{Nothing,Int}}(nothing)

function _window_from_schedule(specification::AbstractString,D::Int)
    pairs=Tuple{Int,Int}[]
    for raw_entry in split(specification,',')
        entry=strip(raw_entry)
        isempty(entry) && continue
        fields=split(entry,':')
        length(fields)==2 || error(
            "ENVIRONMENT_WINDOW_SCHEDULE entries must be maxD:window; got '$entry'")
        maximum_dimension=parse(Int,strip(fields[1]))
        window=parse(Int,strip(fields[2]))
        maximum_dimension>0 || error("Window-schedule maxD must be positive")
        window>0 || error("Scheduled window must be positive")
        push!(pairs,(maximum_dimension,window))
    end
    isempty(pairs) && error("ENVIRONMENT_WINDOW_SCHEDULE is empty")
    issorted(first.(pairs)) || error("ENVIRONMENT_WINDOW_SCHEDULE maxD values must increase")
    length(unique(first.(pairs)))==length(pairs) ||
        error("ENVIRONMENT_WINDOW_SCHEDULE contains duplicate maxD values")
    selected=last(pairs)[2]
    for (maximum_dimension,window) in pairs
        if D<=maximum_dimension
            selected=window
            break
        end
    end
    return selected
end

function _configured_environment_window(D::Int)
    raw=strip(get(ENV,"ENVIRONMENT_WINDOW_SITES","0"))
    if lowercase(raw) in ("schedule","adaptive")
        specification=get(ENV,"ENVIRONMENT_WINDOW_SCHEDULE","")
        isempty(strip(specification)) && error(
            "ENVIRONMENT_WINDOW_SITES=$raw requires ENVIRONMENT_WINDOW_SCHEDULE")
        return _window_from_schedule(specification,D)
    end
    window=parse(Int,raw)
    window>=0 || error("ENVIRONMENT_WINDOW_SITES must be nonnegative")
    return window
end

function _set_environment_window_for_dimension!(D::Int)
    D>0 || error("Bond dimension must be positive when selecting an environment window")
    window=_configured_environment_window(D)
    _active_environment_window_dimension[]=D
    _active_environment_window_sites[]=window
    minimum_window=MPI.Allreduce(window,min,comm)
    maximum_window=MPI.Allreduce(window,max,comm)
    minimum_window==maximum_window ||
        error("Resolved environment window differs across MPI ranks")
    rank==root && println("ENVIRONMENT_WINDOW_SELECTION D=",D," window=",window,
        " configured=",get(ENV,"ENVIRONMENT_WINDOW_SITES","0"))
    return window
end

"""Number of two-site bonds retained in the window frozen for this D pass."""
function environment_window_sites()
    active=_active_environment_window_sites[]
    active===nothing || return active
    D=parse(Int,get(ENV,"BOND_DIMENSION","1"))
    return _configured_environment_window(D)
end

"""Use exact sparse boundary anchors to avoid rebuilding every window from a chain end."""
function environment_window_anchors_enabled()
    environment_window_sites()>0 || return false
    return _parse_bool("ENVIRONMENT_WINDOW_ANCHORS",
        get(ENV,"ENVIRONMENT_WINDOW_ANCHORS","false"))
end

function validate_environment_window_mode!()
    window=environment_window_sites()
    minimum_window=MPI.Allreduce(window,min,comm)
    maximum_window=MPI.Allreduce(window,max,comm)
    minimum_window==maximum_window ||
        error("ENVIRONMENT_WINDOW_SITES differs across MPI ranks")
    tiered=Int(_tiered_pingpong_enabled())
    MPI.Allreduce(tiered,min,comm)==MPI.Allreduce(tiered,max,comm) ||
        error("ENV_TIERED_PINGPONG differs across MPI ranks")
    if tiered==1
        window==0 || error("ENV_TIERED_PINGPONG requires no-window mode")
        _stable_term_owners_enabled() ||
            error("ENV_TIERED_PINGPONG requires ENV_STABLE_TERM_OWNERS=true")
        isempty(environment_store_stats().spill_root) &&
            error("ENV_TIERED_PINGPONG requires ENVIRONMENT_SPILL_ROOT")
    end
    window==0 && return nothing
    _parse_bool("ENV_ROLLING_RELEASE",get(ENV,"ENV_ROLLING_RELEASE","true")) ||
        error("Windowed environments require ENV_ROLLING_RELEASE=true")
    haskey(ENV,"DE") && error("Windowed environments are not validated with DE energy decomposition")
    actual_ist=(@isdefined IST) ? IST : parse(Int,get(ENV,"IST","1"))
    actual_ist==1 ||
        error("Windowed environments currently require IST=1")
    anchors=Int(environment_window_anchors_enabled())
    MPI.Allreduce(anchors,min,comm)==MPI.Allreduce(anchors,max,comm) ||
        error("ENVIRONMENT_WINDOW_ANCHORS differs across MPI ranks")
    stable=Int(_stable_term_owners_enabled())
    MPI.Allreduce(stable,min,comm)==MPI.Allreduce(stable,max,comm) ||
        error("ENV_STABLE_TERM_OWNERS differs across MPI ranks")
    transient_bytes=_window_transient_limit_bytes()
    MPI.Allreduce(transient_bytes,min,comm)==MPI.Allreduce(transient_bytes,max,comm) ||
        error("ENV_WINDOW_TRANSIENT_GB differs across MPI ranks")
    transient_bytes>0 && stable==0 && error(
        "ENV_WINDOW_TRANSIENT_GB requires ENV_STABLE_TERM_OWNERS=true")
    verify_owners=Int(_parse_bool("ENV_WINDOW_VERIFY_OWNERSHIP",
        get(ENV,"ENV_WINDOW_VERIFY_OWNERSHIP","false")))
    MPI.Allreduce(verify_owners,min,comm)==MPI.Allreduce(verify_owners,max,comm) ||
        error("ENV_WINDOW_VERIFY_OWNERSHIP differs across MPI ranks")
    _parse_bool("ENERGY_ONLY",get(ENV,"ENERGY_ONLY","false")) &&
        error("Windowed environments are not yet validated for ENERGY_ONLY")
    if @isdefined cal_excited
        cal_excited==true && error("Windowed environments are not validated for excited-state DMRG")
    end
    return nothing
end

const WINDOW_ANCHOR_DIRECTORY = "__window_anchors__"
const _window_live_anchors = Dict(:right=>Set{Int}(),:left=>Set{Int}())
const _window_environment_site_updates = Ref(0)
const _window_prepare_calls = Ref(0)
const _window_anchor_snapshots = Ref(0)
const _window_anchor_restores = Ref(0)

function _reset_window_work_counters!(direction::Symbol)
    isempty(_window_live_anchors[direction]) ||
        error("Cannot begin $direction pass with live anchors: $(_window_live_anchors[direction])")
    _window_environment_site_updates[]=0
    _window_prepare_calls[]=0
    _window_anchor_snapshots[]=0
    _window_anchor_restores[]=0
    return nothing
end

function _register_window_anchor_plan!(direction::Symbol,positions::Set{Int})
    isempty(_window_live_anchors[direction]) ||
        error("Stale $direction window-anchor session: $(_window_live_anchors[direction])")
    union!(_window_live_anchors[direction],positions)
    return nothing
end

function _consume_window_anchor!(direction::Symbol,position::Int)
    position in _window_live_anchors[direction] ||
        error("Unplanned or already consumed $direction window anchor at cut $position")
    delete!(_window_live_anchors[direction],position)
    return nothing
end

function _finish_window_anchor_session!(rev::Bool)
    direction=rev ? :left : :right
    isempty(_window_live_anchors[direction]) ||
        error("Unconsumed $direction window anchors: $(sort!(collect(_window_live_anchors[direction])))")
    if rank==root && environment_window_sites()>0
        println("WINDOW_WORK direction=",rev ? "reverse" : "forward",
            " window=",environment_window_sites(),
            " anchors=",environment_window_anchors_enabled(),
            " prepare_calls=",_window_prepare_calls[],
            " environment_site_updates=",_window_environment_site_updates[],
            " anchor_snapshots=",_window_anchor_snapshots[],
            " anchor_restores=",_window_anchor_restores[])
    end
    return nothing
end

function _window_anchor_key(direction::Symbol,position::Int,kind::AbstractString)
    direction in (:left,:right) || error("Invalid window-anchor direction: $direction")
    return joinpath(tmp_file,WINDOW_ANCHOR_DIRECTORY,string(direction),string(position),kind)
end

function _right_mini_environment_entries(position::Int)
    terms_e=Vector{Vector{Int64}}()
    for term in terms_keys
        if term[1]<position && term[2]>position
            terms_e=vcat(terms_e,[[term[2],term[3]]])
        elseif term[2]==position
            terms_e=vcat(terms_e,[[term[2],term[3]]])
        end
    end
    terms_e=unique(terms_e)
    return [(owner=_mini_environment_owner(:right,entry[1],entry[end],ii),
        active=joinpath(tmp_file,string(hash([entry[1],[entry[end],1,position]]))))
        for (ii,entry) in enumerate(terms_e)]
end

function _left_mini_environment_entries(position::Int)
    terms_e=Vector{Vector{Int64}}()
    for term in terms_keys
        if position==term[1]
            terms_e=vcat(terms_e,[[term[1],term[3]]])
        elseif term[1]<position && term[2]>position
            terms_e=vcat(terms_e,[[term[1],term[3]]])
        end
    end
    terms_e=unique(terms_e)
    return [(owner=_mini_environment_owner(:left,entry[1],entry[end],ii),
        active=joinpath(tmp_file,string(hash([entry[1],[entry[end],1,position]]))))
        for (ii,entry) in enumerate(terms_e)]
end

function _tiered_pingpong_enabled()
    return _parse_bool("ENV_TIERED_PINGPONG",
        get(ENV,"ENV_TIERED_PINGPONG","false"))
end

function _relocate_environment_key!(key::AbstractString,tier::Symbol)
    environment_backing_exists(key) || return (mode=:missing,bytes=0,path="")
    started=time_ns()
    result=environment_relocate_backing!(key,tier)
    segment=tier==:primary ? :environment_promote : :environment_demote
    moved=result.mode in (:promoted,:demoted) ? result.bytes : 0
    add_segment_timing!(segment,(time_ns()-started)/1.0e9;bytes=moved)
    return result
end

function _relocate_environment_position!(direction::Symbol,position::Int,tier::Symbol)
    L_num=parameter["L"][1]*parameter["L"][2]
    1<=position<=L_num || return nothing
    entries=direction==:right ? _right_mini_environment_entries(position) :
        direction==:left ? _left_mini_environment_entries(position) :
        error("Invalid environment direction: $direction")
    for entry in entries
        entry.owner==rank || continue
        _relocate_environment_key!(entry.active,tier)
    end
    if rank==root
        _relocate_environment_key!(joinpath(tmp_file,
            string(hash([1,position]))),tier)
    end
    if direction==:right && rank==1%sizeofmpi
        _relocate_environment_key!(joinpath(tmp_file,
            string(hash([2,position]))),tier)
    end
    return nothing
end

"""
Rotate the no-window environment tiers at a globally quiescent Krylov boundary.

The same-side frontier just consumed by this direction is demoted for the next
direction, while the next opposite-side frontier is promoted.  A barrier before
and after relocation prevents dynamic H-action workers from observing both
backings during an atomic cross-filesystem copy-and-publish.
"""
function rotate_environment_tiers_after_krylov!(rev::Bool,i::Int)
    _tiered_pingpong_enabled() || return nothing
    environment_window_sites()==0 ||
        error("ENV_TIERED_PINGPONG is only validated in no-window mode")
    isempty(environment_store_stats().spill_root) &&
        error("ENV_TIERED_PINGPONG requires ENVIRONMENT_SPILL_ROOT")
    MPI.Barrier(comm)
    started=time_ns()
    if rev
        _relocate_environment_position!(:right,i+2,:spill)
        _relocate_environment_position!(:left,i-2,:primary)
    else
        _relocate_environment_position!(:left,i-1,:spill)
        _relocate_environment_position!(:right,i+3,:primary)
    end
    MPI.Barrier(comm)
    add_segment_timing!(:environment_tier_wait,(time_ns()-started)/1.0e9)
    return nothing
end

const ENVIRONMENT_EPOCH_MANIFEST="environment_epoch.toml"

function _checkpoint_content_sha(directory::AbstractString,L_num::Int)
    resolved=_resolve_checkpoint_directory(directory)
    manifest=validate_checkpoint(resolved,L_num)
    manifest===nothing && error("Tiered ping-pong requires a checkpoint manifest")
    metadata=get(manifest,"metadata",Dict())
    haskey(metadata,"checkpoint_sha256") ||
        error("Checkpoint manifest lacks checkpoint_sha256: $resolved")
    return String(metadata["checkpoint_sha256"])
end

function _environment_hamiltonian_fingerprint()
    records=String[]
    for term in terms_keys
        push!(records,join(term,",")*"="*repr(parameter[terms[term]]))
    end
    for key in sort!(collect(keys(parameter));by=string)
        key in values(terms) && continue
        value=parameter[key]
        value isa Number || value isa AbstractString || value isa AbstractVector || continue
        push!(records,string(key)*"="*repr(value))
    end
    return bytes2hex(SHA.sha256(join(records,"\n")))
end

function _write_environment_epoch!(state::AbstractString,direction::AbstractString,
        checkpoint_sha::AbstractString)
    state in ("ready","in_progress") || error("Invalid environment epoch state: $state")
    direction in ("left","right") || error("Invalid environment direction: $direction")
    path=joinpath(tmp_file,ENVIRONMENT_EPOCH_MANIFEST)
    data=Dict(
        "schema"=>1,
        "state"=>state,
        "available_direction"=>direction,
        "checkpoint_sha256"=>String(checkpoint_sha),
        "hamiltonian_fingerprint"=>_environment_hamiltonian_fingerprint(),
        "mpi_ranks"=>sizeofmpi,
        "stable_term_owners"=>_stable_term_owners_enabled(),
        "updated_utc"=>string(Dates.now(Dates.UTC)),
    )
    temporary=path*".tmp.$(getpid()).$(time_ns())"
    try
        open(temporary,"w") do io
            TOML.print(io,data;sorted=true)
        end
        mv(temporary,path;force=true)
    finally
        isfile(temporary) && rm(temporary;force=true)
    end
    return path
end

function begin_environment_epoch!(rev::Bool,checkpoint_directory::AbstractString)
    _tiered_pingpong_enabled() || return nothing
    L_num=parameter["L"][1]*parameter["L"][2]
    expected_direction=rev ? "left" : "right"
    checkpoint_sha=_checkpoint_content_sha(checkpoint_directory,L_num)
    epoch_error=nothing
    if rank==root
        try
            path=joinpath(tmp_file,ENVIRONMENT_EPOCH_MANIFEST)
            isfile(path) || error("Missing environment epoch manifest: $path")
            epoch=TOML.parsefile(path)
            get(epoch,"state","")=="ready" ||
                error("Environment epoch is not ready; rebuild environments")
            get(epoch,"available_direction","")==expected_direction ||
                error("Environment direction mismatch: expected $expected_direction")
            get(epoch,"checkpoint_sha256","")==checkpoint_sha ||
                error("Environment/checkpoint identity mismatch")
            get(epoch,"hamiltonian_fingerprint","")==_environment_hamiltonian_fingerprint() ||
                error("Environment/Hamiltonian fingerprint mismatch")
            Int(get(epoch,"mpi_ranks",-1))==sizeofmpi ||
                error("Environment MPI-rank count mismatch")
            _write_environment_epoch!("in_progress",expected_direction,checkpoint_sha)
        catch err
            epoch_error=sprint(showerror,err)
        end
    end
    epoch_error=MPI.bcast(rank==root ? epoch_error : nothing,root,comm)
    epoch_error===nothing || error(epoch_error)
    return checkpoint_sha
end

function finish_environment_epoch!(rev::Bool,checkpoint_directory::AbstractString)
    _tiered_pingpong_enabled() || return nothing
    L_num=parameter["L"][1]*parameter["L"][2]
    checkpoint_sha=_checkpoint_content_sha(checkpoint_directory,L_num)
    epoch_error=nothing
    if rank==root
        try
            _write_environment_epoch!("ready",rev ? "right" : "left",checkpoint_sha)
        catch err
            epoch_error=sprint(showerror,err)
        end
    end
    epoch_error=MPI.bcast(rank==root ? epoch_error : nothing,root,comm)
    epoch_error===nothing || error(epoch_error)
    return checkpoint_sha
end

function _publish_mini_environment_entries!(entries)
    for entry in entries
        entry.owner==rank || continue
        _window_transient_publish!(entry.active)
    end
    return nothing
end

function _audit_window_entry_owners!(entries,direction::Symbol,position::Int)
    _parse_bool("ENV_WINDOW_VERIFY_OWNERSHIP",
        get(ENV,"ENV_WINDOW_VERIFY_OWNERSHIP","false")) || return nothing
    local_owned=count(entry->entry.owner==rank,entries)
    total_owned=MPI.Allreduce(local_owned,+,comm)
    total_owned==length(entries) || error(
        "Window owner partition mismatch at $direction anchor $position: "*
        "owned=$total_owned entries=$(length(entries))")
    invalid_local=count(entry->!(0<=entry.owner<sizeofmpi),entries)
    invalid_total=MPI.Allreduce(invalid_local,+,comm)
    invalid_total==0 || error("Invalid window owner rank at $direction anchor $position")
    rank==root && println("WINDOW_OWNER_AUDIT direction=",direction,
        " position=",position," entries=",length(entries)," status=PASS")
    return nothing
end

function _future_forward_anchor_positions(first_bond::Int,window::Int,L_num::Int)
    anchors=Set{Int}()
    start=first_bond+window
    while start<=L_num-1
        high=min(L_num-1,start+window-1)
        keep_lo=start+2
        keep_hi=min(high+2,L_num)
        keep_lo<=keep_hi && push!(anchors,keep_hi)
        start+=window
    end
    return anchors
end

function _future_reverse_anchor_positions(high_bond::Int,window::Int)
    anchors=Set{Int}()
    high=high_bond-window
    while high>=1
        low=max(1,high-window+1)
        keep_lo=max(1,low-1)
        keep_hi=high-1
        keep_lo<=keep_hi && push!(anchors,keep_lo)
        high-=window
    end
    return anchors
end

function _snapshot_right_environment_anchor!(position::Int,A,A2;transient::Bool=false)
    started=time_ns()
    local_bytes=0
    local_hardlinks=0
    local_copies=0
    if rank==root
        tensor_save(A,_window_anchor_key(:right,position,"combined"))
    end
    if rank==1%sizeofmpi
        A2===nothing && error("Missing right norm environment at anchor $position")
        tensor_save(A2,_window_anchor_key(:right,position,"norm"))
    end
    entries=_right_mini_environment_entries(position)
    _audit_window_entry_owners!(entries,:right,position)
    for entry in entries
        entry.owner==rank || continue
        cloned=transient ? _window_transient_snapshot!(entry.active,
            _window_anchor_key(:right,position,basename(entry.active))) :
            environment_clone_backing!(entry.active,
                _window_anchor_key(:right,position,basename(entry.active)))
        local_bytes+=cloned.bytes
        local_hardlinks+=Int(cloned.mode==:hardlink)
        local_copies+=Int(cloned.mode==:copy)
    end
    environment_cache_clear!()
    local_seconds=(time_ns()-started)/1.0e9
    maximum_seconds=MPI.Allreduce(local_seconds,max,comm)
    total_bytes=MPI.Allreduce(local_bytes,+,comm)
    total_hardlinks=MPI.Allreduce(local_hardlinks,+,comm)
    total_copies=MPI.Allreduce(local_copies,+,comm)
    add_segment_timing!(:window_anchor_snapshot,local_seconds;bytes=local_bytes)
    _window_anchor_snapshots[]+=1
    rank==root && println("WINDOW_ANCHOR_SNAPSHOT direction=right position=",position,
        " max_rank_seconds=",maximum_seconds," mini_bytes=",total_bytes,
        " hardlinks=",total_hardlinks," copies=",total_copies)
    return nothing
end

function _snapshot_left_environment_anchor!(position::Int,A;transient::Bool=false)
    started=time_ns()
    local_bytes=0
    local_hardlinks=0
    local_copies=0
    if rank==root
        tensor_save(A,_window_anchor_key(:left,position,"combined"))
    end
    entries=_left_mini_environment_entries(position)
    _audit_window_entry_owners!(entries,:left,position)
    for entry in entries
        entry.owner==rank || continue
        cloned=transient ? _window_transient_snapshot!(entry.active,
            _window_anchor_key(:left,position,basename(entry.active))) :
            environment_clone_backing!(entry.active,
                _window_anchor_key(:left,position,basename(entry.active)))
        local_bytes+=cloned.bytes
        local_hardlinks+=Int(cloned.mode==:hardlink)
        local_copies+=Int(cloned.mode==:copy)
    end
    environment_cache_clear!()
    local_seconds=(time_ns()-started)/1.0e9
    maximum_seconds=MPI.Allreduce(local_seconds,max,comm)
    total_bytes=MPI.Allreduce(local_bytes,+,comm)
    total_hardlinks=MPI.Allreduce(local_hardlinks,+,comm)
    total_copies=MPI.Allreduce(local_copies,+,comm)
    add_segment_timing!(:window_anchor_snapshot,local_seconds;bytes=local_bytes)
    _window_anchor_snapshots[]+=1
    rank==root && println("WINDOW_ANCHOR_SNAPSHOT direction=left position=",position,
        " max_rank_seconds=",maximum_seconds," mini_bytes=",total_bytes,
        " hardlinks=",total_hardlinks," copies=",total_copies)
    return nothing
end

function _restore_right_environment_anchor!(position::Int;transient::Bool=false)
    started=time_ns()
    local_bytes=0
    A=nothing
    A2=nothing
    if rank==root
        anchor=_window_anchor_key(:right,position,"combined")
        active=joinpath(tmp_file,string(hash([1,position])))
        moved=environment_move_backing!(anchor,active)
        local_bytes+=moved.bytes
        A=tensor_load(active)
    end
    if rank==1%sizeofmpi
        anchor=_window_anchor_key(:right,position,"norm")
        active=joinpath(tmp_file,string(hash([2,position])))
        moved=environment_move_backing!(anchor,active)
        local_bytes+=moved.bytes
        A2=tensor_load(active)
    end
    entries=_right_mini_environment_entries(position)
    _audit_window_entry_owners!(entries,:right,position)
    for entry in entries
        entry.owner==rank || continue
        anchor=_window_anchor_key(:right,position,basename(entry.active))
        moved=environment_move_backing!(anchor,entry.active)
        local_bytes+=moved.bytes
        transient && _window_transient_import!(entry.active)
    end
    environment_cache_clear!()
    local_seconds=(time_ns()-started)/1.0e9
    maximum_seconds=MPI.Allreduce(local_seconds,max,comm)
    total_bytes=MPI.Allreduce(local_bytes,+,comm)
    add_segment_timing!(:window_anchor_restore,local_seconds;bytes=local_bytes)
    _window_anchor_restores[]+=1
    _consume_window_anchor!(:right,position)
    rank==root && println("WINDOW_ANCHOR_RESTORE direction=right position=",position,
        " max_rank_seconds=",maximum_seconds," moved_bytes=",total_bytes)
    return A,A2
end

function _restore_left_environment_anchor!(position::Int;transient::Bool=false)
    started=time_ns()
    local_bytes=0
    A=nothing
    if rank==root
        anchor=_window_anchor_key(:left,position,"combined")
        active=joinpath(tmp_file,string(hash([1,position])))
        moved=environment_move_backing!(anchor,active)
        local_bytes+=moved.bytes
        A=tensor_load(active)
    end
    entries=_left_mini_environment_entries(position)
    _audit_window_entry_owners!(entries,:left,position)
    for entry in entries
        entry.owner==rank || continue
        anchor=_window_anchor_key(:left,position,basename(entry.active))
        moved=environment_move_backing!(anchor,entry.active)
        local_bytes+=moved.bytes
        transient && _window_transient_import!(entry.active)
    end
    environment_cache_clear!()
    local_seconds=(time_ns()-started)/1.0e9
    maximum_seconds=MPI.Allreduce(local_seconds,max,comm)
    total_bytes=MPI.Allreduce(local_bytes,+,comm)
    add_segment_timing!(:window_anchor_restore,local_seconds;bytes=local_bytes)
    _window_anchor_restores[]+=1
    _consume_window_anchor!(:left,position)
    rank==root && println("WINDOW_ANCHOR_RESTORE direction=left position=",position,
        " max_rank_seconds=",maximum_seconds," moved_bytes=",total_bytes)
    return A
end

"""
Build only the right environments needed by forward bonds `first_bond:last_bond`.

Term environments outside the retained window are propagated once and deleted
immediately.  The combined/full environment is kept in memory during the
boundary-to-window contraction and serialized only for retained cuts.
"""
function prepare_right_environment_window!(save_file::String,first_bond::Int,last_bond::Int;
        initialize_anchors::Bool=false)
    L_num=parameter["L"][1]*parameter["L"][2]
    1<=first_bond<=last_bond<=L_num-1 ||
        error("Invalid forward environment window $first_bond:$last_bond")
    keep_lo=first_bond+2
    keep_hi=min(last_bond+2,L_num)
    keep_lo<=keep_hi || return nothing
    transient=_window_transient_enabled()
    transient && _window_transient_reset!()

    mini_seconds=0.0
    full_seconds=0.0
    combined_seconds=0.0
    publish_seconds=0.0
    prepare_started=time_ns()
    progress_interval=max(1,parse(Int,get(ENV,"ENV_WINDOW_PROGRESS_SITES","16")))
    use_anchors=environment_window_anchors_enabled()
    anchor_positions=initialize_anchors && use_anchors ?
        _future_forward_anchor_positions(first_bond,environment_window_sites(),L_num) : Set{Int}()
    initialize_anchors && use_anchors && _register_window_anchor_plan!(:right,anchor_positions)
    A=nothing
    A2=nothing
    first_position=L_num
    if use_anchors && !initialize_anchors
        A,A2=_restore_right_environment_anchor!(keep_hi;transient=transient)
        first_position=keep_hi-1
    end
    total_positions=max(first_position-keep_lo+1,0)
    processed=0
    for position in first_position:-1:keep_lo
        _window_environment_site_updates[]+=1
        ttn=tensor_load(joinpath(save_file,string(position)))
        mini_start=time_ns()
        update_blocks_right_mini(ttn,position;transient=transient)
        transient || MPI.Barrier(comm)
        mini_seconds+=(time_ns()-mini_start)/1.0e9

        full_start=time_ns()
        combined_start=time_ns()
        if position==L_num
            @tensor A[-1,-2]:=ttn[-1,1,2]*terms_onsite[L_num][1,3]*conj(ttn[-2,3,2])
            if rank==1%sizeofmpi
                @tensor A2[-1,-2]:=ttn[-1,1,2]*conj(ttn[-2,1,2])
            end
        else
            if A===nothing
                @tensor A[-1,-2]:=TensorMap(zeros,space(ttn,1),space(ttn,1))[-1,-2]
            end
            sites=find_terms(position,direction=1)
            A=right_block_add(A,position,ttn,sites;transient=transient)
            delete_full_term_blocks!(sites;direction=:right,transient=transient)
            if rank==1%sizeofmpi
                @tensor A2[-1,-2]:=A2[1,2]*ttn[-1,3,1]*conj(ttn[-2,3,2])
            end
        end
        combined_seconds+=(time_ns()-combined_start)/1.0e9
        publish_start=time_ns()
        if position<=keep_hi
            transient && _publish_mini_environment_entries!(
                _right_mini_environment_entries(position))
            rank==0 && tensor_save(A,joinpath(tmp_file,string(hash([1,position]))))
            rank==1%sizeofmpi && tensor_save(A2,joinpath(tmp_file,string(hash([2,position]))))
        end
        publish_seconds+=(time_ns()-publish_start)/1.0e9
        if position in anchor_positions
            _snapshot_right_environment_anchor!(position,A,A2;transient=transient)
        end
        full_seconds+=(time_ns()-full_start)/1.0e9

        stale=position+1
        if stale<=L_num && stale>keep_hi
            delete_blocks_right_mini(stale;transient=transient)
        end
        processed+=1
        if rank==root && initialize_anchors &&
                (processed==1 || processed%progress_interval==0 ||
                 processed==total_positions || position in anchor_positions)
            println("WINDOW_PREP_PROGRESS direction=forward position=",position,
                " processed=",processed," total=",total_positions,
                " anchors_done=",_window_anchor_snapshots[],
                " elapsed_seconds=",(time_ns()-prepare_started)/1.0e9)
        end
    end
    add_segment_timing!(:initial_mini_blocks,mini_seconds)
    add_segment_timing!(:initial_full_blocks,full_seconds)
    add_segment_timing!(:window_prepass_mini,mini_seconds)
    add_segment_timing!(:window_prepass_combined,combined_seconds)
    add_segment_timing!(:window_retained_publish,publish_seconds)
    transient && _window_transient_finish!(:right)
    environment_cache_clear!()
    MPI.Barrier(comm)
    return nothing
end

"""Build only the left environments needed by reverse bonds `low_bond:high_bond`."""
function prepare_left_environment_window!(save_file::String,low_bond::Int,high_bond::Int;
        initialize_anchors::Bool=false)
    L_num=parameter["L"][1]*parameter["L"][2]
    1<=low_bond<=high_bond<=L_num-1 ||
        error("Invalid reverse environment window $low_bond:$high_bond")
    keep_lo=max(1,low_bond-1)
    keep_hi=high_bond-1
    keep_hi>=1 || return nothing
    transient=_window_transient_enabled()
    transient && _window_transient_reset!()

    mini_seconds=0.0
    full_seconds=0.0
    combined_seconds=0.0
    publish_seconds=0.0
    prepare_started=time_ns()
    progress_interval=max(1,parse(Int,get(ENV,"ENV_WINDOW_PROGRESS_SITES","16")))
    use_anchors=environment_window_anchors_enabled()
    anchor_positions=initialize_anchors && use_anchors ?
        _future_reverse_anchor_positions(high_bond,environment_window_sites()) : Set{Int}()
    initialize_anchors && use_anchors && _register_window_anchor_plan!(:left,anchor_positions)
    A=nothing
    first_position=1
    if use_anchors && !initialize_anchors
        A=_restore_left_environment_anchor!(keep_lo;transient=transient)
        first_position=keep_lo+1
    end
    total_positions=max(keep_hi-first_position+1,0)
    processed=0
    for position in first_position:keep_hi
        _window_environment_site_updates[]+=1
        ttn=tensor_load(joinpath(save_file,string(position)))
        mini_start=time_ns()
        update_blocks_left_mini(ttn,position;transient=transient)
        transient || MPI.Barrier(comm)
        mini_seconds+=(time_ns()-mini_start)/1.0e9

        full_start=time_ns()
        combined_start=time_ns()
        if position==1
            @tensor A[-1,-2]:=ttn[2,1,-1]*terms_onsite[1][1,3]*conj(ttn[2,3,-2])
        else
            if A===nothing
                @tensor A[-1,-2]:=TensorMap(zeros,space(ttn,3),space(ttn,3))[-1,-2]
            end
            sites=find_terms(position)
            A=left_block_add(A,position,ttn,sites;transient=transient)
            delete_full_term_blocks!(sites;direction=:left,transient=transient)
        end
        combined_seconds+=(time_ns()-combined_start)/1.0e9
        publish_start=time_ns()
        if position>=keep_lo
            transient && _publish_mini_environment_entries!(
                _left_mini_environment_entries(position))
            rank==0 && tensor_save(A,joinpath(tmp_file,string(hash([1,position]))))
        end
        publish_seconds+=(time_ns()-publish_start)/1.0e9
        if position in anchor_positions
            _snapshot_left_environment_anchor!(position,A;transient=transient)
        end
        full_seconds+=(time_ns()-full_start)/1.0e9

        stale=position-1
        if stale>=1 && stale<keep_lo
            delete_blocks_left_mini(stale;transient=transient)
        end
        processed+=1
        if rank==root && initialize_anchors &&
                (processed==1 || processed%progress_interval==0 ||
                 processed==total_positions || position in anchor_positions)
            println("WINDOW_PREP_PROGRESS direction=reverse position=",position,
                " processed=",processed," total=",total_positions,
                " anchors_done=",_window_anchor_snapshots[],
                " elapsed_seconds=",(time_ns()-prepare_started)/1.0e9)
        end
    end
    add_segment_timing!(:initial_mini_blocks,mini_seconds)
    add_segment_timing!(:initial_full_blocks,full_seconds)
    add_segment_timing!(:window_prepass_mini,mini_seconds)
    add_segment_timing!(:window_prepass_combined,combined_seconds)
    add_segment_timing!(:window_retained_publish,publish_seconds)
    transient && _window_transient_finish!(:left)
    environment_cache_clear!()
    MPI.Barrier(comm)
    return nothing
end

function prepare_environment_window!(rev::Bool,save_file::String,bond::Int;
        initialize_anchors::Bool=false)
    window=environment_window_sites()
    window>0 || return nothing
    if @isdefined cal_excited
        cal_excited==true && error("Windowed environments are not validated for excited-state DMRG")
    end
    L_num=parameter["L"][1]*parameter["L"][2]
    _window_prepare_calls[]+=1
    started=time_ns()
    if rev
        low=max(1,bond-window+1)
        rank==root && println("ENVIRONMENT_WINDOW direction=reverse bonds=",low,":",bond)
        prepare_left_environment_window!(save_file,low,bond;
            initialize_anchors=initialize_anchors)
    else
        high=min(L_num-1,bond+window-1)
        rank==root && println("ENVIRONMENT_WINDOW direction=forward bonds=",bond,":",high)
        prepare_right_environment_window!(save_file,bond,high;
            initialize_anchors=initialize_anchors)
    end
    add_segment_timing!(:window_prepare,(time_ns()-started)/1.0e9)
    return nothing
end



function MPS_sweep(b_save_file::String,save_file::String;D::Int64=100,ite::Int64=1,rev::Bool=false)
    _set_environment_window_for_dimension!(D)
    validate_environment_window_mode!()
    reset_segment_timings!()
    _reset_window_work_counters!(rev ? :left : :right)
    sweep_started=time_ns()
    checkpoint_prepare_started=time_ns()
    ge=[0.0]
    L_num=parameter["L"][1]*parameter["L"][2]
    krylov_dim=parse(Int,get(ENV,"KRYLOV_DIM","3"))
    krylov_tol=parse(Float64,get(ENV,"KRYLOV_TOL","1e-14"))
    truncation_cutoff=parse(Float64,get(ENV,"TRUNCATION_CUTOFF","0.0"))
    truncation_cutoff>=0 || error("TRUNCATION_CUTOFF must be nonnegative")
    if rank==root
        source=(isdir(save_file) && _checkpoint_tensor_path(save_file,1)!==nothing) ? save_file : b_save_file
        source_identity=checkpoint_identity(source,L_num)
        work_file=begin_checkpoint_generation(source,save_file,L_num;
            generation="sweep-$(ite)-D$(D)",
            metadata=Dict("source_sha256"=>source_identity,"bond_dimension"=>D))
    else
        work_file=nothing
    end
    work_file=MPI.bcast(work_file,root,comm)
    read_file=_resolve_checkpoint_directory(b_save_file)
    MPI.Barrier(comm)
    begin_environment_epoch!(rev,read_file)
    add_segment_timing!(:checkpoint_prepare,
        (time_ns()-checkpoint_prepare_started)/1.0e9)
    truncs=zeros(Float32,L_num-1)
    benchmark_limit_raw=strip(get(ENV,"BENCHMARK_MAX_BONDS",""))
    benchmark_max_bonds=isempty(benchmark_limit_raw) ? L_num-1 : parse(Int,benchmark_limit_raw)
    1<=benchmark_max_bonds<=L_num-1 ||
        error("BENCHMARK_MAX_BONDS must be in 1:$(L_num-1), got $benchmark_max_bonds")
    bounded_benchmark=benchmark_max_bonds<L_num-1
    processed_bonds=0
    if bounded_benchmark && rank==root
        println("BOUNDED_SWEEP direction=",rev ? "reverse" : "forward",
            " max_bonds=",benchmark_max_bonds,
            " total_bonds=",L_num-1)
    end
    PA_NUM=1
    re_time=1
    finish_sig=1
    ist=1
    if ite==1
        if @isdefined IST
            ist=IST
        end
    end
    if rev
        ttn1=tensor_load(read_file*"/"*string(L_num))
    else
        ttn2=tensor_load(read_file*"/"*string(ist))
    end

    for k in ist:L_num-1
        i=k
        if rev
            i=L_num-k
        end
        window=environment_window_sites()
        if window>0 && (k-ist)%window==0
            prepare_environment_window!(rev,read_file,i;
                initialize_anchors=(k==ist && environment_window_anchors_enabled()))
        end
        st=time()
        trunc_err=0
        if rev
            ttn2=ttn1
            ttn1=tensor_load(read_file*"/"*string(i))
        else
            ttn1=ttn2
            ttn2=tensor_load(read_file*"/"*string(i+1))
        end
        load_time=time()
        
        if rank==0
            write_data_txt("out/"*out_put_file,string(domain(ttn1)))
            write_data_txt("out/"*out_put_file,string(size_T(ttn1)))
        end

        ## solving eigenvalue problem
        krylov_started=time_ns()
        ge,gv,re_time,PA_NUM=eigsh_lanczos(ttn1,ttn2,i,krylov_dim,1,krylov_tol)
        add_segment_timing!(:krylov,(time_ns()-krylov_started)/1.0e9)
        release_consumed_environment!(rev,i)
        rotate_environment_tiers_after_krylov!(rev,i)
        #meminfo_julia()

        eigen_time=time()

        mode=svd_execution_mode()
        truncation=truncation_cutoff>0 ?
            (truncdim(D) & truncerr(sqrt(truncation_cutoff))) : truncdim(D)
        svd_started=time_ns()
        if mode==:serial
            U,S,V,trunc_err=tsvd(gv[1],(1,2),(3,4);trunc=truncation)
        else
            gv[1]=permute(gv[1],(1,2),(3,4))
            U,S,V,trunc_err=tsvd_MPI(gv[1];trunc=truncation)
        end
        add_segment_timing!(:svd,(time_ns()-svd_started)/1.0e9)

        svd_time=time()
        finish_sig==1

        if rank==0
            ## calculate spectrum data
            if (haskey(ENV,"Spectrum"))
                ss=sort(diag(convert(Array,S)),rev=true)
                en=sum([-(se*se)*log(se*se) for se in ss])
                Ham_save(ss,"EE/"*out_put_file*"/Spectrum_"*string(D)*"/"*string(i))
                write_data_txt("EE/"*out_put_file*"_EE",string([i,D,en]))
            end

            if rev
                ttn2=permute(V,(1,2),(3,))
                @tensor A[-1,-2;-3]:=U[-1,-2,1]*S[1,-3]
                A=A/norm(A)
                ttn1=A
            else
                ttn1=permute(U,(1,2),(3,))
                @tensor A[-1,-2;-3]:=S[-1,1]*V[1,-2,-3]
                A=A/norm(A)
                ttn2=A
            end
            truncs[i]=trunc_err^2

            tensor_save(ttn1,work_file*"/"*string(i))
            finish_sig=tensor_save(ttn2,work_file*"/"*string(i+1))


        end
        MPI.Barrier(comm)

        # ##  read from file
        finish_sig=MPI.bcast(rank==0 ? finish_sig : nothing,root,comm)
        if finish_sig==0
            ttn1=tensor_load(work_file*"/"*string(i))
            ttn2=tensor_load(work_file*"/"*string(i+1))
        else
            throw("save wave-function error")
        end

        ## update tensor in ttn  ### MPI.bcast is slow for multi-node running.
        # ttn1=MPI.bcast(rank==0 ? ttn1 : nothing,root,comm)
        # ttn2=MPI.bcast(rank==0 ? ttn2 : nothing,root,comm)

        update_time=time()
        
        ## update mini blocks of a single hamiltonian term
        block_started=time_ns()
        update_mini(rev,i,ttn1,ttn2)
        MPI.Barrier(comm)
        # Mini-block updates may overwrite a key cached by another rank.
        # Clear every rank before the combined block consumes those updates.
        environment_cache_clear!()
        MPI.Barrier(comm)

        ## update super blocks
        updata_super(rev,i,ttn1,ttn2)
        if environment_window_sites()>0
            release_propagated_environment!(rev,i)
        end
        add_segment_timing!(:block_update,(time_ns()-block_started)/1.0e9)
        if i%5==0
            GC.gc()
        end
        MPI.Barrier(comm)
        environment_cache_clear!()
        MPI.Barrier(comm)

        if rank==0
            if !(isfile("time/"*out_put_file*".txt"))
                write_data_txt("time/"*out_put_file,"ite, load_time, eigen_time, svd_time, update_MPS_time,update_block_time,computing_part,MAX_MPI")
            end
            write_data_txt("time/"*out_put_file,string([i,Float32.(load_time-st),Float32.(eigen_time-load_time),Float32.(svd_time-eigen_time),Float32.(update_time-svd_time),Float32.(time()-update_time),Float32.(re_time),Float32.(PA_NUM)]))
            write_data_txt("out/"*out_put_file, string([i,trunc_err,ge[1]/(L_num),Float32.(time()-st)]))
        end
        processed_bonds+=1
        processed_bonds>=benchmark_max_bonds && break
    end
    MPI.Barrier(comm)
    _finish_window_anchor_session!(rev)
    window_cleanup_started=time_ns()
    cleanup_window_sweep_frontier!(rev)
    add_segment_timing!(:window_cleanup,
        (time_ns()-window_cleanup_started)/1.0e9)
    checkpoint_finalize_started=time_ns()
    if rank==root
        write_checkpoint_manifest(work_file,L_num;
            generation="sweep-$(ite)",
            metadata=Dict(
                "managed_by"=>"DMRG-checkpoint-v2",
                "source_sha256"=>source_identity,
                "bond_dimension"=>D,
                "reverse"=>rev,
                "krylov_dim"=>krylov_dim,
                "krylov_tol"=>krylov_tol,
                "truncation_cutoff_squared_norm"=>truncation_cutoff,
                "reported_truncation_error"=>"squared discarded singular-value norm",
                "svd_mode"=>string(svd_execution_mode()),
                "environment_window_sites"=>environment_window_sites(),
                "environment_window_anchors"=>environment_window_anchors_enabled(),
                "partial_direction"=>bounded_benchmark,
                "processed_bonds"=>processed_bonds,
                "direction_bonds"=>L_num-1,
            ))
        commit_checkpoint_generation(save_file,work_file,L_num)
    end
    MPI.Barrier(comm)
    finish_environment_epoch!(rev,save_file)
    add_segment_timing!(:checkpoint_finalize,
        (time_ns()-checkpoint_finalize_started)/1.0e9)
    add_segment_timing!(:sweep_total,(time_ns()-sweep_started)/1.0e9)
    report_segment_timings!("sweep-$(ite)-$(rev ? "reverse" : "forward")",comm;root=root)
    if bounded_benchmark && rank==root
        println("BOUNDED_SWEEP_COMPLETE direction=",rev ? "reverse" : "forward",
            " processed_bonds=",processed_bonds,
            " total_bonds=",L_num-1)
    end
    return ge[1]/L_num,maximum(truncs)
end

function _parse_bool(name::AbstractString,value::AbstractString)
    normalized=lowercase(strip(value))
    normalized in ("1","true","yes","on") && return true
    normalized in ("0","false","no","off") && return false
    error("$name must be one of 1/0, true/false, yes/no, or on/off; got '$value'")
end

"""
Resolve SVD execution explicitly. SVD_MODE=auto|serial|mpi is preferred.
For compatibility SVD_MPI=true now means MPI (rather than the old inverted
presence test); false means serial.
"""
function svd_execution_mode()
    if haskey(ENV,"SVD_MODE")
        mode=Symbol(lowercase(strip(ENV["SVD_MODE"])))
        mode in (:auto,:serial,:mpi) || error("SVD_MODE must be auto, serial, or mpi")
    elseif haskey(ENV,"SVD_MPI")
        mode=_parse_bool("SVD_MPI",ENV["SVD_MPI"]) ? :mpi : :serial
    else
        mode=:auto
    end
    mode==:auto && return sizeofmpi>1 ? :mpi : :serial
    mode==:mpi && sizeofmpi<2 && error("MPI SVD requires at least two MPI ranks; use SVD_MODE=serial")
    return mode
end

function tsvd_MPI(t::TensorMap;
                trunc::TensorKit.TruncationScheme = TensorKit.NoTruncation(),
                p::Real = 2,
                alg::Union{TensorKit.SVD, TensorKit.SDD} = SDD())
    S = spacetype(t)
    I = sectortype(t)
    scalar_type = eltype(t)
    real_type = typeof(real(zero(scalar_type)))
    # `storagetype(t)` is the flat backing Vector type in TensorKit 0.14,
    # whereas SectorDict entries are dense matrix blocks.
    Udata = TensorKit.SectorDict{I, Matrix{scalar_type}}()
    Σmdata = TensorKit.SectorDict{I, Matrix{real_type}}()
    Σdata = TensorKit.SectorDict{I, Vector{real_type}}()
    Vdata = TensorKit.SectorDict{I, Matrix{scalar_type}}()
    dims = TensorKit.SectorDict{sectortype(t), Int}()
    if isempty(blocksectors(t))
        W = S(dims)
        truncerr = zero(real(Float64))
        return TensorMap(Udata, codomain(t)←W), TensorMap(Σmdata, W←W),
                    TensorMap(Vdata, W←domain(t)), truncerr
    end

    data_sectors=blocksectors(t)
    # TensorKit exposes QN blocks as reshaped/subarray views.  The distributed
    # LAPACK call mutates its input and requires an owned dense matrix on every
    # rank, so materialize each block explicitly.
    data=Matrix{Float64}[Matrix(blocks(t)[i]) for i in data_sectors]
    data_svd=_svd_pa(data;alg=alg)
    if rank==0
        for i in 1:length(data_svd)
            Udata[data_sectors[i]] = data_svd[i][1]
            Vdata[data_sectors[i]] = data_svd[i][3]
            Σdata[data_sectors[i]] = data_svd[i][2]
            dims[data_sectors[i]] = length(data_svd[i][2])
        end

        if !isa(trunc, TensorKit.NoTruncation)
            truncdims = TensorKit._compute_truncdim(Σdata,trunc,p)
            truncerr = TensorKit._compute_truncerr(Σdata,truncdims,p)
            for c in blocksectors(t)
                truncdim = truncdims[c]
                if truncdim != 0
                    if truncdim != dims[c]
                        Udata[c] = Udata[c][:, 1:truncdim]
                        Vdata[c] = Vdata[c][1:truncdim, :]
                        Σdata[c] = Σdata[c][1:truncdim]
                    end
                else
                    delete!(Udata, c)
                    delete!(Vdata, c)
                    delete!(Σdata, c)
                end
            end
            dims = TensorKit.SectorDict{I,Int}(c=>d for (c,d) in truncdims if d>0)
            W = S(dims)
        else
            W = S(dims)
            if length(domain(t)) == 1 && domain(t)[1] ≅ W
                W = domain(t)[1]
            elseif length(codomain(t)) == 1 && codomain(t)[1] ≅ W
                W = codomain(t)[1]
            end
            truncerr = abs(zero(Float64))
        end
        for (c, Σ) in Σdata
            Σmdata[c] = copyto!(similar(Σ, length(Σ), length(Σ)), Diagonal(Σ))
        end
        return TensorMap(Udata, codomain(t)←W), TensorMap(Σmdata, W←W),
                TensorMap(Vdata, W←domain(t)), truncerr
    else
        return 0,0,0,0
    end
end


function _svd_pa(data::Vector{Matrix{Float64}};MAX_NUM::Int64=100,alg::Union{TensorKit.SVD, TensorKit.SDD} = SDD())
    nworkers = sizeofmpi - 1
    send_mesg = Array{Int64}(undef, 1)
    recv_mesg = Array{Int64}(undef, 1)
    out_main=Vector{Vector{Array}}(undef,length(data))
    recv_mesg_rank=Vector{Array}(undef,4)
    N=length(data)
    position = Array{Int64}(undef, 1)
    #BLAS.set_num_threads(Threads.nthreads())

    if rank==0
        idx_recv = 0
        idx_sent = 1
        status_workers = ones(nworkers).*-1
        send_buffers = [zeros(Int64,1) for _ in 1:nworkers]
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
            send_buffers[dst][1] = idx_sent
            MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
            idx_sent += 1
            status_workers[dst] = 1
        end
        while idx_recv != N
            for dst in 1:nworkers
                if status_workers[dst] == 1
                    ismessage = MPI.Iprobe(comm; source=dst, tag=dst+MAX_NUM)
                    if ismessage
                        if haskey(ENV,"SVD_SAVE")
                            position=MPI.recv(comm; source=dst, tag=dst+MAX_NUM)
                            out_main[position[1]]=tensor_load(joinpath(tmp_file,string(hash(position))))
                        else
                            recv_mesg_rank=MPI.recv(comm; source=dst, tag=dst+MAX_NUM)
                            out_main[recv_mesg_rank[4][1]]=recv_mesg_rank[1:3]
                        end
                        idx_recv += 1
                        if idx_sent <= N
                            send_buffers[dst][1] = idx_sent
                            MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
                            idx_sent += 1
                            status_workers[dst] = 1
                        else
                            status_workers[dst] = -1
                        end
                    end
                end
            end
        end

        for dst in 1:nworkers
            # Termination message to worker
            send_buffers[dst][1] = -1
            MPI.Send(send_buffers[dst], comm; dest=dst, tag=dst+MAX_NUM)
        end
    else
        while true
            ismessage = MPI.Iprobe(comm; source=root, tag=rank+MAX_NUM)
            if ismessage
                # Receives message
                MPI.Recv!(recv_mesg, comm; source=root, tag=rank+MAX_NUM)
                # Termination message from root
                if recv_mesg[1] == -1
                    break
                end
                # Apply function (SVD for a mini block) to array
                lapack_started=time_ns()
                U, Σ, V = TensorKit.MatrixAlgebra.svd!(data[recv_mesg[1]], alg)
                add_segment_timing!(:svd_lapack_active,(time_ns()-lapack_started)/1.0e9;
                    bytes=sizeof(data[recv_mesg[1]]))
                if haskey(ENV,"SVD_SAVE")
                    Ham_save([U, Σ, V],joinpath(tmp_file,string(hash(recv_mesg))))
                    MPI.send(recv_mesg, comm; dest=root, tag=rank+MAX_NUM)
                else
                    MPI.send([U, Σ, V,recv_mesg], comm; dest=root, tag=rank+MAX_NUM)
                end
            end
        end
    end
    #BLAS.set_num_threads(1)
    return out_main
end

function transfer_type_wavefunciton()
    b_save_file="TNc/"*out_put_file*"_"*ENV["INPUT_BOND_DIMENSION"]
    save_file="TNc/"*out_put_file*"_"*ENV["BOND_DIMENSION"]
    Len=parameter["L"][1]*parameter["L"][2]
    if rank==0
        p_bar = Progress(Len,desc=out_put_file*"_transfer",barlen=30 ,barglyphs=BarGlyphs("[=> ]"),showspeed=true)
        for i in 1:parameter["L"][1]*parameter["L"][2]
            ttn=1.0*tensor_load(b_save_file*"/"*string(i))
            tensor_save(ttn,save_file*"/"*string(i))
            ProgressMeter.next!(p_bar)
        end
    end
    return 0
end

function checkpoint_path_for_dimension(D::Integer)
    default="TNc/"*out_put_file*"_"*string(D)
    haskey(ENV,"OUTPUT_MPS_DIR") || return default
    initial_D=parse(Int,ENV["BOND_DIMENSION"])
    return D==initial_D ? abspath(ENV["OUTPUT_MPS_DIR"]) : abspath(ENV["OUTPUT_MPS_DIR"])*"_D$(D)"
end

function _validate_model_checkpoint(directory::AbstractString,Len::Integer)
    return validate_checkpoint(directory,Len;
        physical_space=V[1],left_boundary=V_in[1],right_boundary=V_out[1])
end

function initial_MPS_wavefunction()
    save_file=checkpoint_path_for_dimension(parse(Int,ENV["BOND_DIMENSION"]))
    b_save_file="TNc/"*out_put_file*"_"*ENV["INPUT_BOND_DIMENSION"]
    Len=parameter["L"][1]*parameter["L"][2]
    start_file=get(ENV,"START_MPS_DIR","")
    if !isempty(start_file)
        if rank==root
            _validate_model_checkpoint(start_file,Len)
            start_identity=checkpoint_identity(start_file,Len)
            if isdir(save_file)
                _parse_bool("RESUME",get(ENV,"RESUME","false")) ||
                    error("OUTPUT_MPS_DIR already exists; set RESUME=true only for the same source")
                manifest=_validate_model_checkpoint(save_file,Len)
                manifest===nothing && error("Cannot resume output without a checkpoint manifest")
                get(manifest["metadata"],"source_sha256","")==start_identity ||
                    error("RESUME source hash does not match existing output")
            else
                work_file=begin_checkpoint_generation(start_file,save_file,Len;
                    generation="START_MPS_DIR-import",
                    metadata=Dict("source_sha256"=>start_identity))
                if norm(tensor_load(work_file*"/1"))>1.01
                    norm_TN2(1,work_file,L=parameter["L"])
                end
                write_checkpoint_manifest(work_file,Len;
                    generation="START_MPS_DIR-import",
                    metadata=Dict("managed_by"=>"DMRG-checkpoint-v2","source_sha256"=>start_identity))
                commit_checkpoint_generation(save_file,work_file,Len)
                _validate_model_checkpoint(save_file,Len)
            end
        end
        MPI.Barrier(comm)
        environment_cache_clear!()
        MPI.Barrier(comm)
    elseif isdir(b_save_file)
        if rank==root && !isdir(save_file)
            source_identity=checkpoint_identity(b_save_file,Len)
            work_file=begin_checkpoint_generation(b_save_file,save_file,Len;
                generation="bond-dimension-import",
                metadata=Dict("source_sha256"=>source_identity))
            write_checkpoint_manifest(work_file,Len;
                generation="bond-dimension-import",
                metadata=Dict("managed_by"=>"DMRG-checkpoint-v2","source_sha256"=>source_identity))
            commit_checkpoint_generation(save_file,work_file,Len)
        end
        MPI.Barrier(comm)
    else
        if rank==root
            isdir(save_file) && error("Output checkpoint exists but has no usable starting state: $save_file")
            work_file=begin_empty_checkpoint_generation(save_file;generation="initial")
            ttn=TN_initial()
            for i in 1:3
                ttn=initial_ttn(ttn)
            end
            #ttn=norm_TN(1,ttn,L=L)
            for i in 1:length(ttn)
                tensor_save(ttn[i],work_file*"/"*string(i))
            end
            write_checkpoint_manifest(work_file,Len;
                generation="initial",
                metadata=Dict("managed_by"=>"DMRG-checkpoint-v2"))
            commit_checkpoint_generation(save_file,work_file,Len)
        end
    end
    MPI.Barrier(comm)
    if rank==root
        _validate_model_checkpoint(save_file,Len)
    end
    MPI.Barrier(comm)
    return 0
end

"""Delete completed full-term contractions after every rank consumed them."""
function delete_full_term_blocks!(sites;direction::Symbol,transient::Bool=false)
    if haskey(ENV,"DE") ||
            !_parse_bool("ENV_ROLLING_RELEASE",get(ENV,"ENV_ROLLING_RELEASE","true"))
        environment_cache_clear!()
        return 0
    end
    transient || MPI.Barrier(comm)
    for (ii,term) in enumerate(sites)
        if _full_term_owner(direction,term,ii)==rank
            _window_env_delete(joinpath(tmp_file,
                string(hash([[term[1],term[2]],[term[3],1,0]]))),transient)
        end
    end
    transient || MPI.Barrier(comm)
    environment_cache_clear!()
    return 0
end

"""
Release environments that cannot be used by any later bond in this pass.

At bond (i,i+1), a forward pass consumes right environments at i+2, while a
reverse pass consumes left environments at i-1.  Release happens only after
the complete Krylov solve, never between Hamiltonian applications.
"""
function release_consumed_environment!(rev::Bool,i::Int64)
    if !_parse_bool("ENV_ROLLING_RELEASE",get(ENV,"ENV_ROLLING_RELEASE","true"))
        environment_cache_clear!()
        MPI.Barrier(comm)
        return 0
    end
    if @isdefined cal_excited
        if cal_excited==true
            # Excited-state overlap environments have a different last-use
            # schedule; the ground-state rolling proof does not apply.
            environment_cache_clear!()
            MPI.Barrier(comm)
            return 0
        end
    end
    L_num=parameter["L"][1]*parameter["L"][2]
    position=rev ? i-1 : i+2
    if 1<=position<=L_num
        if rev
            delete_blocks_left_mini(position)
        else
            delete_blocks_right_mini(position)
        end
        if rank==root
            write_zero(joinpath(tmp_file,string(hash([1,position]))))
        end
        if !rev && rank==1%sizeofmpi
            write_zero(joinpath(tmp_file,string(hash([2,position]))))
        end
        if @isdefined cal_excited
            if cal_excited==true && rank==1%sizeofmpi
                key=rev ? [3,position] : [4,position]
                write_zero(joinpath(tmp_file,string(hash(key))))
            end
        end
    end
    environment_cache_clear!()
    MPI.Barrier(comm)
    return 0
end

"""
In window mode, release the same-side predecessor only after it has been
propagated by update_mini/updata_super.  This prevents newly generated left
(forward) or right (reverse) environments from accumulating across the chain.
"""
function release_propagated_environment!(rev::Bool,i::Int64)
    environment_window_sites()>0 || return 0
    L_num=parameter["L"][1]*parameter["L"][2]
    position=rev ? i+2 : i-1
    if 1<=position<=L_num
        if rev
            delete_blocks_right_mini(position)
        else
            delete_blocks_left_mini(position)
        end
        if rank==root
            write_zero(joinpath(tmp_file,string(hash([1,position]))))
        end
    end
    environment_cache_clear!()
    MPI.Barrier(comm)
    return 0
end

"""Remove the final same-side frontier left after the last bond of a windowed pass."""
function cleanup_window_sweep_frontier!(rev::Bool)
    environment_window_sites()>0 || return 0
    L_num=parameter["L"][1]*parameter["L"][2]
    # The endpoint update in update_mini closes the final crossing terms, but
    # there is no following site update that would normally consume them.
    endpoint_terms=rev ? find_terms(1;direction=1) : find_terms(L_num)
    delete_full_term_blocks!(endpoint_terms;direction=rev ? :right : :left)
    position=rev ? 2 : L_num-1
    if rev
        delete_blocks_right_mini(position)
    else
        delete_blocks_left_mini(position)
    end
    if rank==root
        write_zero(joinpath(tmp_file,string(hash([1,position]))))
    end
    environment_cache_clear!()
    MPI.Barrier(comm)
    return 0
end

function initial_step_MPS(p::Int64=1)
    validate_profile_segments!(comm)
    _set_environment_window_for_dimension!(parse(Int,ENV["BOND_DIMENSION"]))
    validate_environment_window_mode!()
    reset_segment_timings!()
    initial_started=time_ns()
    save_file=_resolve_checkpoint_directory(checkpoint_path_for_dimension(parse(Int,ENV["BOND_DIMENSION"])))
    environment_cache_clear!()
    MPI.Barrier(comm)
    if rank==0
        cleanup_environment_backing!()
        println("ENVIRONMENT_STORE_ROOT=",tmp_file)
        println("ENVIRONMENT_STORE_SPILL_ROOT=",environment_store_stats().spill_root)
        println("ENVIRONMENT_LOCAL_CACHE_BYTES=",environment_store_stats().cache_limit_bytes)
        println("ENVIRONMENT_PRIMARY_CAP_BYTES=",
            environment_store_stats().primary_cap_bytes)
        println("ENV_TIERED_PINGPONG=",_tiered_pingpong_enabled())
        println("CHANNEL_SCHEDULE_STATS ",channel_schedule_stats())
        println("H_ACTION_SCHEDULER=",
            lowercase(strip(get(ENV,"H_ACTION_SCHEDULER","dynamic_workers"))))
    end

    window=environment_window_sites()
    if window==0
        MPI.Barrier(comm)
        mini_started=time_ns()
        initial_blocks_mini(save_file;p=p)
        add_segment_timing!(:initial_mini_blocks,(time_ns()-mini_started)/1.0e9)
        MPI.Barrier(comm)

        full_started=time_ns()
        initial_blocks(save_file;p=p)
        add_segment_timing!(:initial_full_blocks,(time_ns()-full_started)/1.0e9)
        MPI.Barrier(comm)
    else
        p==1 || error("Fresh reverse window initialization is not supported")
        rank==root && println("ENVIRONMENT_WINDOW_SITES=",window,
            " initial full-chain environment construction deferred to each window")
    end
    if _tiered_pingpong_enabled()
        if rank==root
            checkpoint_sha=_checkpoint_content_sha(save_file,
                parameter["L"][1]*parameter["L"][2])
            _write_environment_epoch!("ready",p==1 ? "right" : "left",checkpoint_sha)
        end
        MPI.Barrier(comm)
    end
    add_segment_timing!(:initial_total,(time_ns()-initial_started)/1.0e9)
    report_segment_timings!(window==0 ? "initial-full-chain" : "initial-window-deferred",
        comm;root=root)
    if @isdefined cal_excited
        if cal_excited==true
            initial_excited_blocks(save_file)
        end
    end
    MPI.Barrier(comm)
    return 0
end

function cleanup_environment_store!()
    println("ENVIRONMENT_STORE_STATS rank=",rank," ",environment_store_stats())
    environment_cache_clear!()
    MPI.Barrier(comm)
    if rank==root
        cleanup_environment_backing!(remove_roots=true)
    end
    MPI.Barrier(comm)
    return 0
end

"""Evaluate <psi|H|psi>/<psi|psi> without changing the checkpoint."""
function current_mps_energy(;position::Int=1)
    Len=parameter["L"][1]*parameter["L"][2]
    1 <= position < Len || error("ENERGY_ONLY_POSITION must be in 1:$(Len-1)")
    save_file=_resolve_checkpoint_directory(
        checkpoint_path_for_dimension(parse(Int,ENV["BOND_DIMENSION"])))
    if environment_window_sites()>0
        prepare_right_environment_window!(save_file,position,position)
    end
    ttn1=tensor_load(joinpath(save_file,string(position)))
    ttn2=tensor_load(joinpath(save_file,string(position+1)))
    @tensor vector[-1,-2,-3,-4] := ttn1[-1,-2,1]*ttn2[1,-3,-4]
    hvector,_,_=enviroment_general(vector,position)
    energy=nothing
    norm2=nothing
    if rank==root
        norm2=real(dot(vector,vector))
        norm2>0 || error("MPS two-site center has zero norm")
        energy=real(dot(vector,hvector))/norm2
        println("CURRENT_MPS_NORM2=",norm2)
        println("CURRENT_MPS_ENERGY=",energy)
        println("CURRENT_MPS_ENERGY_PER_SITE=",energy/Len)
    end
    energy=MPI.bcast(energy,root,comm)
    MPI.Barrier(comm)
    return energy
end

function calculate_energy_parts(rev,D)
    energy_file="Hamiltonain/"*out_put_file*"_"*string(D)
    save_file=checkpoint_path_for_dimension(D)
    energy=Dict{Vector{Int64},Float64}()
    m=terms_keys
    energy_rank=zeros(length(m))
    if rev==false
        if rank==0
            ttn=tensor_load(save_file*"/"*string(parameter["L"][1]*parameter["L"][2]))
            @tensor B[-1,-2]:=ttn[-1,1,2]*conj(ttn[-2,1,2])
            tensor_save(B,tmp_file*"/"*string(hash([2,parameter["L"][1]*parameter["L"][2]])))
            for i in parameter["L"][1]*parameter["L"][2]-1:-1:1
                ttn=tensor_load(save_file*"/"*string(i))
                @tensor B[-1,-2]:=B[1,2]*ttn[-1,3,1]*conj(ttn[-2,3,2])
                tensor_save(B,tmp_file*"/"*string(hash([2,i])))
            end
        end
        MPI.Barrier(comm)
        for i in 1:length(m)
            if i%sizeofmpi==rank
                A=tensor_load(tmp_file*"/"*string(hash([[m[i][1],m[i][2]],[m[i][3],1,0]])))
                if m[i][2]+1<=parameter["L"][1]*parameter["L"][2]
                    B=tensor_load(tmp_file*"/"*string(hash([2,m[i][2]+1])))
                    @tensor A[]:=A[1,2]*B[1,2]
                else
                    @tensor A[]:=A[1,1]
                end
                energy_rank[i]=values(A.data)[1][1]
            end
        end
    else
        if rank==0
            ttn=tensor_load(save_file*"/"*string(1))
            @tensor B[-1,-2]:=ttn[1,2,-1]*conj(ttn[1,2,-2])
            tensor_save(B,tmp_file*"/"*string(hash([-2,1])))
            for i in 2:parameter["L"][1]*parameter["L"][2]
                ttn=tensor_load(save_file*"/"*string(i))
                @tensor B[-1,-2]:=B[1,2]*ttn[1,3,-1]*conj(ttn[2,3,-2])
                tensor_save(B,tmp_file*"/"*string(hash([-2,i])))
            end
        end
        MPI.Barrier(comm)
        for i in 1:length(m)
            if i%sizeofmpi==rank
                A=tensor_load(tmp_file*"/"*string(hash([[m[i][1],m[i][2]],[m[i][3],1,0]])))
                if m[i][1]-1>0
                    B=tensor_load(tmp_file*"/"*string(hash([-2,m[i][1]-1])))
                    @tensor A[]:=A[1,2]*B[1,2]
                else
                    @tensor A[]:=A[1,1]
                end
                energy_rank[i]=values(A.data)[1][1]
            end
        end
    end
    energys=MPI.Reduce(energy_rank,+,root,comm)
    out=0
    if rank==0
        for i in 1:length(m)
            energy[m[i]]=energys[i]
        end
        Ham_save(energy,energy_file)
        out=0
        for i in collect(keys(energy))
            if terms[i]==ENV["DE"]
                out=out+energy[i]
            end
        end
    end
    return out/(parameter["L"][1]*parameter["L"][2])
end




function MPS()
    mkpath("out/"*parameter["model"])
    mkpath("out_short/"*parameter["model"])
    mkpath("time/"*parameter["model"])
    EE=[0.0,100]

    D=parse(Int64,ENV["BOND_DIMENSION"])
    save_file=checkpoint_path_for_dimension(D)
    ham_file="Hamiltonain/"*out_put_file
    mkpath(ham_file)
    b_save_file=save_file
    out_put_short_file="out_short/"*out_put_file
    i=0

    end_bond=parse(Int64,ENV["STOP_BOND_DIMENSION"])
    ##iteration of MPS

    if rank==0
        write_txt(ham_file*"/bond_dimension",ENV["STOP_BOND_DIMENSION"])
    end
    MPI.Barrier(comm)

    while D<=read_txt(ham_file*"/bond_dimension")
        dE=0.0
        i=i+1
        MPI.Barrier(comm)
        if i%2==0
            rev=true
        else
            rev=false
        end
        st=time()
        E,trunc_err=MPS_sweep(b_save_file,save_file;D=D,ite=i,rev=rev)
        if haskey(ENV,"DE")
            dE=calculate_energy_parts(rev,D)
        end
        MPI.Barrier(comm)
        if rank==0
            if !(isfile(out_put_short_file*".txt"))
                write_data_txt(out_put_short_file,"sweep, bond-dimension, trunc_err, E per site, dE, time")
            end
            write_data_txt(out_put_short_file,string([i,D,trunc_err,E,dE,Float32.(time()-st)]))
        end
        EE[1]=EE[2]
        EE[2]=E
        b_save_file=save_file

        if EE[1]-EE[2]<1e-5 && D<2000
            if D<400
                D=400
            elseif D<1000
                D=1000
            elseif D<2000
                D=2000
            end
        elseif EE[1]-EE[2]<1e-5 && D<10000
            if D<10000
                D=D+2000
            end
        elseif EE[1]-EE[2]<1e-5 && D<20000
            if D<20000
                D=D+5000
            end
        elseif EE[1]-EE[2]<5e-6 && D<60000
            if D<60000
                D=D+10000
            end
        elseif EE[1]-EE[2]<5e-6 && D<160000
            if D<160000
                D=D+20000
            else
                break
            end
        end
        D=MPI.bcast(rank==0 ? D : nothing,root,comm)
        save_file=checkpoint_path_for_dimension(D)
    end
    cleanup_environment_store!()
    return 0
end

function MPS_SP()
    mkpath("out/"*parameter["model"])
    mkpath("out_short/"*parameter["model"])
    mkpath("time/"*parameter["model"])
    EE=[0.0,100]
    L=parameter["L"]
    D=parse(Int64,ENV["BOND_DIMENSION"])
    save_file=checkpoint_path_for_dimension(D)
    ham_file="Hamiltonain/"*out_put_file
    mkpath(ham_file)
    b_save_file=save_file
    out_put_short_file="out_short/"*out_put_file
    i=0
    end_bond=parse(Int64,ENV["STOP_BOND_DIMENSION"])
    first_reverse=_parse_bool("FIRST_PASS_REVERSE",get(ENV,"FIRST_PASS_REVERSE","false"))
    ##iteration of MPS

    if rank==0
        write_txt(ham_file*"/bond_dimension",ENV["STOP_BOND_DIMENSION"])
    end
    MPI.Barrier(comm)
    while (D<=read_txt(ham_file*"/bond_dimension")) && (i<length(SWEEP)) 
        dE=0.0
        i=i+1
        MPI.Barrier(comm)
        rev=isodd(i) ? first_reverse : !first_reverse
        st=time()
        E,trunc_err=MPS_sweep(b_save_file,save_file,D=D,ite=i,rev=rev)
        if haskey(ENV,"DE")
            dE=calculate_energy_parts(rev,D)
        end
        MPI.Barrier(comm)
        if rank==0
            if !(isfile(out_put_short_file*".txt"))
                write_data_txt(out_put_short_file,"sweep, bond-dimension, trunc_err, E per site, dE, time")
            end
            write_data_txt(out_put_short_file,string([i,D,trunc_err,E,dE,Float32.(time()-st)]))
        end
        EE[1]=EE[2]
        EE[2]=E
        b_save_file=save_file
        if i<length(SWEEP)
            D=MPI.bcast(rank==0 ? SWEEP[i+1] : nothing,root,comm)
        end
        save_file=checkpoint_path_for_dimension(D)
    end
    cleanup_environment_store!()
    return 0
end
