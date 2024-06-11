import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch import nn

def arc_margin_product(
        input: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False) -> torch.Tensor:
    
    # Normalize input and weight
    input = F.normalize(input, dim=1)
    weight = F.normalize(weight, dim=1)
    
    # Calculate cosine similarity
    cosine = torch.matmul(input, weight.t())
    
    # Calculate phi
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    phi = cosine * cos_m - sine * sin_m
    
    if easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > th, phi, cosine - mm)
    
    N = cosine.size(0)
    targets = targets.view(-1, 1)  # Make sure targets is of shape (N, 1)
    
    # Gather the cosine and phi values corresponding to the target labels
    cosine_of_targets = cosine.gather(1, targets)
    phi_of_targets = phi.gather(1, targets)
    
    # Combine cos and phi using the target labels
    output = cosine.clone()
    output.scatter_(1, targets, phi_of_targets)
    output = output * s
    
    return output

def add_margin_product(input, targets, weight, s=30.0, m=0.40):
    # Normalize input and weight
    input = F.normalize(input, dim=1)
    weight = F.normalize(weight, dim=1)

    # Calculate cosine similarity
    cosine = torch.matmul(input, weight.t())
    phi = cosine - m

    N = cosine.size(0)
    targets = targets.view(-1, 1)  # Make sure targets is of shape (N, 1)

    # Gather the cosine and phi values corresponding to the target labels
    cosine_of_targets = cosine.gather(1, targets)
    phi_of_targets = phi.gather(1, targets)

    # Combine cos and phi using the target labels
    output = cosine.clone()
    output.scatter_(1, targets, phi_of_targets)
    output = output * s

    return output

def sphere_product(input, targets, weight, m=4, base=1000.0, gamma=0.12, power=1, LambdaMin=5.0, iter_count=0):
    # Normalize input and weight
    input = F.normalize(input, dim=1)
    weight = F.normalize(weight, dim=1)

    mlambda = [
        lambda x: x ** 0,
        lambda x: x ** 1,
        lambda x: 2 * x ** 2 - 1,
        lambda x: 4 * x ** 3 - 3 * x,
        lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
    ]

    def lambda_func(iter_count):
        return max(LambdaMin, base * (1 + gamma * iter_count) ** (-1 * power))

    lamb = lambda_func(iter_count)
    iter_count += 1

    # Calculate cosine similarity
    cos_theta = torch.matmul(input, weight.t())
    cos_theta = cos_theta.clamp(-1, 1)
    cos_m_theta = mlambda[m](cos_theta)
    theta = cos_theta.data.acos()
    k = (m * theta / math.pi).floor()
    phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
    NormOfFeature = torch.norm(input, 2, 1)

    N = cos_theta.size(0)
    targets = targets.view(-1, 1)  # Make sure targets is of shape (N, 1)

    # Gather the cosine and phi values corresponding to the target labels
    cosine_of_targets = cos_theta.gather(1, targets)
    phi_of_targets = phi_theta.gather(1, targets)

    # Combine cos and phi using the target labels
    output = cos_theta.clone()
    output.scatter_(1, targets, phi_of_targets)
    output = output * NormOfFeature.view(-1, 1)

    return output

# def main():
#     input = torch.randn(10, 512).cuda()  # 假设的输入
#     targets = torch.randint(0, 100, (10,)).cuda()  # 假设的标签
#     in_features = 512
#     out_features = 100

#     # 初始化权重
#     weight = torch.FloatTensor(out_features, in_features).cuda()
#     nn.init.xavier_uniform_(weight)

#     # 计算结果
#     arc_output = arc_margin_product(input, targets, weight)
#     add_output = add_margin_product(input, targets, weight)
#     sphere_output = sphere_product(input, targets, weight)

#     print("Arc Margin Product Output:\n", arc_output)
#     # print("Add Margin Product Output:\n", add_output)
#     # print("Sphere Product Output:\n", sphere_output)

# if __name__ == "__main__":
#     main()