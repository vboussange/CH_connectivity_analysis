using SparseArrays
using LinearAlgebra

function CommuteCostToTarget(P::SparseMatrixCSC, C::SparseMatrixCSC, target::Int)
    #Computes the directed commute-cost distances from all nodes to a target node.
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -target: the target node id
    #Output: the directed commute-cost vector
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    if target<1 || target>n
        throw(DimensionMismatch("The target node is not valid"))
    end
    Pc=sum(P.*C,dims=2);
    Pc[target]=0;
    IP=I-P;
    IP[target,:]=zeros(n);
    IP[target,target]=1;
    return IP\Pc;
end

function CommuteCostFull(P::SparseMatrixCSC,C::SparseMatrixCSC)
    #Computes the entire directed commute-cost distance matrix.
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    #Output: the directed commute-cost matrix
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    Pc=sum(P.*C,dims=2);
    IP=I-P;
    D=zeros(n,n);
    vt=0;
    v=zeros(n);
    for target in Base.OneTo(n)
        if target>1
            IP[target-1,:]=v;
            Pc[target-1]=vt;
        end
        vt=Pc[target];
        v=IP[target,:];
        Pc[target]=0;
        IP[target,:]=zeros(n);
        IP[target,target]=1;
        D[:,target]=IP\Pc;
    end
    return D;
end

function StationaryDistribution(P::SparseMatrixCSC)
    #Computes the stationary distribution of a random walk following the transition probability matrix
    #Input: the transition probability matrix P
    #Output: the stationary distribution of the random walk
    n = LinearAlgebra.checksquare(P);
    PI=P'-I;
    PI[1,:]=ones(n);
    v=[1;zeros(n-1)];
    return PI\v;
end

function QWeightedNodeBetweenness(P::SparseMatrixCSC,C::SparseMatrixCSC,qˢ::Vector,qᵗ::Vector)
    #Computes the Q-weighted betweenness for every node
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -qˢ: the source quality vector
    # -qᵗ: the target quality vector
    #Ouput: the node q-weighted betweenness vector
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    if length(qˢ)!=n
        throw(DimensionMismatch("The length of the vector qˢ does not match with the number of nodes"))
    end
    if length(qᵗ)!=n
        throw(DimensionMismatch("The length of the vector qᵗ does not match with the number of nodes"))
    end
    p=StationaryDistribution(P);
    Z=inv(Matrix(I-P).+p');
    H=(diag(Z)'.-Z)./p';
    bet=zeros(n);
    for t in Base.OneTo(n)
        bet.+=((Z.-Z[t,:]'+p'.*H[:,t])'*qˢ).*qᵗ[t];
    end
    return bet;
end

function QWeightedNodeBetweenness_Subset(P::SparseMatrixCSC,C::SparseMatrixCSC,qˢ::Vector,qᵗ::Vector,nodelist::Vector)
    #Computes the K-weighted betweenness for a subset of nodes
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -qˢ: the source quality vector
    # -qᵗ: the target quality vector
    # -nodelist: the list of nodes for which the betweenness should be computed
    #Ouput: the vector with the same length as the nodelist, containing the node q-weighted betweennesses
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    if length(qˢ)!=n
        throw(DimensionMismatch("The length of the vector qˢ does not match with the number of nodes"))
    end
    if length(qᵗ)!=n
        throw(DimensionMismatch("The length of the vector qᵗ does not match with the number of nodes"))
    end
    p=StationaryDistribution(P);
    Z=inv(Matrix(I-P).+p');
    H=(diag(Z)'.-Z)./p';
    bet=zeros(length(nodelist));
    for (index,node) in enumerate(nodelist)
        bet[index]=qˢ'*(Z[:,node].-Z[:,node]'+H.*p[node])*qᵗ;
    end
    return bet;
end

function QWeightedEdgeBetweenness(P::SparseMatrixCSC,C::SparseMatrixCSC,qˢ::Vector,qᵗ::Vector)
    #Computes the Q-weighted betweenness for every edge
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -qˢ: the source quality vector
    # -qᵗ: the target quality vector
    #Ouput: the edge q-weighted betweenness matrix
    i,j,pref=findnz(P);
    bet=zeros(length(pref));
    nodebet=QWeightedNodeBetweenness(P,C,qˢ,qᵗ);
    for (edge,node) in enumerate(i)
        bet[edge]=nodebet[node]*pref[edge];
    end
    return sparse(i,j,bet);
end

function KWeightedNodeBetweenness(P::SparseMatrixCSC,C::SparseMatrixCSC,K::AbstractMatrix)
    #Computes the K-weighted betweenness for every node
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -K: the proximity matrix
    #Ouput: the node k-weighted betweenness vector
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    if LinearAlgebra.checksquare(K)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and K do not match"))
    end
    p=StationaryDistribution(P);
    Z=inv(Matrix(I-P).+p');
    H=(diag(Z)'.-Z)./p';
    bet=zeros(n);
    for t in Base.OneTo(n)
        bet.+=((Z.-Z[t,:]'+p'.*H[:,t])'*K[:,t]);
    end
    return bet;
end

function KWeightedNodeBetweenness_Subset(P::SparseMatrixCSC,C::SparseMatrixCSC,K::AbstractMatrix,nodelist::Vector)
    #Computes the K-weighted betweenness for a subset of nodes
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -K: the proximity matrix
    # -nodelist: the list of nodes for which the betweenness should be computed
    #Ouput: the vector with the same length as the nodelist, containing the node k-weighted betweennesses
    n = LinearAlgebra.checksquare(P);
    if LinearAlgebra.checksquare(C)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and C do not match"))
    end
    if LinearAlgebra.checksquare(K)!=n
        throw(DimensionMismatch("The dimensions of the matrices Pref and K do not match"))
    end
    p=StationaryDistribution(P);
    Z=inv(Matrix(I-P).+p');
    H=(diag(Z)'.-Z)./p';
    bet=zeros(length(nodelist));
    for (index,node) in enumerate(nodelist)
        bet[index]=sum((Z[:,node].-Z[:,node]'+H.*p[node]).*K);
    end
    return bet;
end

function KWeightedEdgeBetweenness(P::SparseMatrixCSC,C::SparseMatrixCSC,K::AbstractMatrix)
    #Computes the K-weighted betweenness for every edge
    #Inputs:
    # -P: the transition probability matrix
    # -C: the transition cost matrix
    # -K: the proximity matrix
    #Ouput: the edge K-weighted betweenness matrix
    i,j,pref=findnz(P);
    bet=zeros(length(pref));
    nodebet=KWeightedNodeBetweenness(P,C,K);
    for (edge,node) in enumerate(i)
        bet[edge]=nodebet[node]*pref[edge];
    end
    return sparse(i,j,bet);
end

