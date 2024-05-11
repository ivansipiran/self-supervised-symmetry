try:
    import open3d
except:
    pass

import torch
import options as options
import util
import torch.optim as optim
from losses import chamfer_distance
from data_handler import get_dataset
from torch.utils.data import DataLoader
from models import SymmetryNetwork, SymmetryNetwork2
import numpy as np


import torch

def get_rot_matrix(Np, angle):
    #Np = torch.squeeze(N)
    #Np = torch.nn.functional.normalize(Np,dim=0)

    angle = angle * np.pi/180

    S = torch.zeros((3,3))
    S[0,1] = -Np[2]
    S[0,2] = Np[1]
    S[1,0] = Np[2]
    S[1,2] = -Np[0]
    S[2,0] = -Np[1]
    S[2,1] = Np[0]

    R = torch.eye(3) + torch.sin(angle)*S + (1-torch.cos(angle))*torch.matmul(S,S)
    return R.unsqueeze(0)

def get_sym_matrix(Np):
    #Np = torch.squeeze(Np)
    #Np = torch.nn.functional.normalize(Np,dim=0)
    #angle = angle * np.pi/180
    
    R = torch.zeros((Np.shape[0],3,3))
    R[:,0,0] = 1-2*Np[:,0]*Np[:,0]
    R[:,0,1] = -2*Np[:,0]*Np[:,1]
    R[:,0,2] = -2*Np[:,0]*Np[:,2]
    R[:,1,0] = -2*Np[:,1]*Np[:,0]
    R[:,1,1] = 1-2*Np[:,1]*Np[:,1]
    R[:,1,2] = -2*Np[:,1]*Np[:,2]
    R[:,2,0] = -2*Np[:,2]*Np[:,0]
    R[:,2,1] = -2*Np[:,2]*Np[:,1]
    R[:,2,2] = 1-2*Np[:,2]*Np[:,2]
    return R

def reflective_loss(N1, N2, N3, d2, device):
    Np1 = torch.squeeze(N1)
    Np2 = torch.squeeze(N2)
    Np3 = torch.squeeze(N3)
    #print(f'Inside Loss - N1: {Np1.shape}')

    #Normalization is per row (first dimension is batch size)
    Np1 = torch.nn.functional.normalize(Np1,dim=1)
    Np2 = torch.nn.functional.normalize(Np2,dim=1)
    Np3 = torch.nn.functional.normalize(Np3,dim=1)

    R1 = get_sym_matrix(Np1).to(device)
    R2 = get_sym_matrix(Np2).to(device)
    R3 = get_sym_matrix(Np3).to(device)

    #print(f'R1: {R1.shape}')
    #print(f'R2: {R2.shape}')
    #print(f'R3: {R3.shape}')
    
    output = d2.clone()
    output = output.transpose(2,1)

    output1 = torch.bmm(output, R1)
    output2 = torch.bmm(output, R2)
    output3 = torch.bmm(output, R3)

    output1 = output1.transpose(2,1)
    output2 = output2.transpose(2,1)
    output3 = output3.transpose(2,1)

    loss1 = chamfer_distance(output1, d2, mse=args.mse) + chamfer_distance(output2, d2, mse=args.mse) + chamfer_distance(output3, d2, mse=args.mse)
    #print(f'Loss1: {loss1.shape}')

    #M = torch.cat((Np1.unsqueeze(1),Np2.unsqueeze(1),Np2.unsqueeze(1)),1)
    M = torch.reshape(torch.cat((Np1,Np2,Np3), 1), (d2.shape[0],3,3))
    MtM = torch.bmm(M, torch.transpose(M, 1,2))

    I = torch.eye(3)
    I = I.reshape((1,3,3))
    I = I.repeat(d2.shape[0],1,1)
    #MtM = torch.matmul(torch.transpose(M, 0,1), M)

    I = torch.eye(M.size(-1), dtype=M.dtype, device = M.device)
    loss2 = torch.linalg.matrix_norm(torch.abs(MtM) - I, ord='fro')
    #print(f'Loss2: {loss2.shape}')
    
    loss = (loss1 + loss2).mean()

    return loss

def rotational_loss(N1, N2, N3, d2, device):
    Np1 = torch.squeeze(N1)
    Np2 = torch.squeeze(N2)
    Np3 = torch.squeeze(N3)
    Np1 = torch.nn.functional.normalize(Np1,dim=0)
    Np2 = torch.nn.functional.normalize(Np2,dim=0)
    Np3 = torch.nn.functional.normalize(Np3,dim=0)

    R11 = get_rot_matrix(Np1,30).to(device)
    R12 = get_rot_matrix(Np1,60).to(device)
    R13 = get_rot_matrix(Np1,90).to(device)

    R21 = get_rot_matrix(Np2, 30).to(device)
    R22 = get_rot_matrix(Np2, 60).to(device)
    R23 = get_rot_matrix(Np2, 90).to(device)

    R31 = get_rot_matrix(Np3, 30).to(device)
    R32 = get_rot_matrix(Np3, 60).to(device)
    R33 = get_rot_matrix(Np3, 90).to(device)

    output = d2.clone()
    output = output.transpose(2,1)

    output11 = torch.bmm(output, R11)
    output12 = torch.bmm(output, R12)
    output13 = torch.bmm(output, R13)

    output21 = torch.bmm(output, R21)
    output22 = torch.bmm(output, R22)
    output23 = torch.bmm(output, R23)

    output31 = torch.bmm(output, R31)
    output32 = torch.bmm(output, R32)
    output33 = torch.bmm(output, R33)

    output11 = output11.transpose(2,1)
    output12 = output12.transpose(2,1)
    output13 = output13.transpose(2,1)

    output21 = output21.transpose(2,1)
    output22 = output22.transpose(2,1)
    output23 = output23.transpose(2,1)

    output31 = output31.transpose(2,1)
    output32 = output32.transpose(2,1)
    output33 = output33.transpose(2,1)


    loss1 = chamfer_distance(output11, d2, mse=args.mse) + chamfer_distance(output12, d2, mse=args.mse) + chamfer_distance(output13, d2, mse=args.mse) 
    loss1 = loss1 + chamfer_distance(output21, d2, mse=args.mse) + chamfer_distance(output22, d2, mse=args.mse) + chamfer_distance(output23, d2, mse=args.mse)
    loss1 = loss1 + chamfer_distance(output31, d2, mse=args.mse) + chamfer_distance(output32, d2, mse=args.mse) + chamfer_distance(output33, d2, mse=args.mse)
    M = torch.cat((Np1.unsqueeze(1),Np2.unsqueeze(1),Np2.unsqueeze(1)),1)
        
    MtM = torch.matmul(torch.transpose(M, 0,1), M)
    I = torch.eye(M.size(-1), dtype=M.dtype, device = M.device)
    loss2 = torch.linalg.matrix_norm(torch.abs(MtM) - I, ord='fro')
    loss = loss1 + loss2

    return loss1, loss2, loss


def train(args):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    #device = torch.device('cpu')
    print(f'device: {device}')

    target_pc: torch.Tensor = util.get_input(args, center=True).unsqueeze(0).permute(0, 2, 1).to(device)
    print(target_pc.shape)

    if 0 < args.max_points < target_pc.shape[2]:
        indx = torch.randperm(target_pc.shape[2])
        target_pc = target_pc[:, :, indx[:args.cut_points]]

    data_loader = get_dataset(args.sampling_mode)(target_pc[0].transpose(0, 1), device, args)
    #util.export_pc(target_pc[0], args.save_path / 'target.xyz',
    #               color=torch.tensor([255, 0, 0]).unsqueeze(-1).expand(3, target_pc.shape[-1]))

    model = SymmetryNetwork2()


    print(f'number of parameters: {util.n_params(model)}')
    #model.initialize_params(args.init_var)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations, eta_min=0.00001)
    model.train()
    train_loader = DataLoader(data_loader, num_workers=0,
                          batch_size=args.batch_size, shuffle=False, drop_last=False)

    for i, (d1, d2) in enumerate(train_loader):
        d1, d2 = d1.to(device), d2.to(device)

        model.train()
        optimizer.zero_grad()
        
        #print(f'Input tensor:-{d2.shape}')
        N1, N2, N3 = model(d2)
        #print(f'Normal output:{N1.shape}')

        loss = reflective_loss(N1, N2, N3, d2,device)
        #loss1, loss2, loss = rotational_loss(N1, N2, N3, d2, device)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            print(f'{args.save_path.name}; iter: {i} / {int(len(data_loader) / args.batch_size)}; Loss: {util.show_truncated(loss.item(), 6)}')
            #print(f'{args.save_path.name}; iter: {i} / {int(len(data_loader) / args.batch_size)}; Loss1: {util.show_truncated(loss1.item(), 6)}; Loss2: {util.show_truncated(loss2.item(), 6)}; Loss: {util.show_truncated(loss.item(), 6)}')

    for i, (d1, d2) in enumerate(train_loader):
        d1, d2 = d1.to(device), d2.to(device)
        d2 = d2[0,:,:]
        d2= d2.unsqueeze(0)
        print(d2.shape)
        N1, N2, N3 = model(d2)
        #Np1 = torch.squeeze(N1)
        #Np2 = torch.squeeze(N2)
        #Np3 = torch.squeeze(N3)
        Np1 = torch.nn.functional.normalize(N1,dim=1)
        Np2 = torch.nn.functional.normalize(N2,dim=1)
        Np3 = torch.nn.functional.normalize(N3,dim=1)

        R1 = get_sym_matrix(Np1).to(device)
        R2 = get_sym_matrix(Np2).to(device)
        R3 = get_sym_matrix(Np3).to(device)
        #R1 = get_rot_matrix(Np1, 30).to(device)
        #R2 = get_rot_matrix(Np2, 30).to(device)
        #R3 = get_rot_matrix(Np3, 30).to(device)

        #x = d2.transpose(2,1)
        #print(f'Input shape:{x.shape}')
        #x1 = torch.bmm(x, R1)
        #x2 = torch.bmm(x, R2)
        #x3 = torch.bmm(x, R3)

        #x1 = x1.transpose(2,1)
        #x2 = x2.transpose(2,1)
        #x3 = x3.transpose(2,1)
        
        np.savetxt('matriz_sym1.txt', R1[0].cpu().detach().numpy())
        np.savetxt('matriz_sym2.txt', R2[0].cpu().detach().numpy())
        np.savetxt('matriz_sym3.txt', R3[0].cpu().detach().numpy())

        #print(f'Matrix: {matrix.shape}')
        #print(sym_matrix)
        #print(x.shape)
        #print(torch.det(sym_matrix))
        #util.export_pc(x[0], 'prueba.xyz')
        break
    
        #if i % args.export_interval == 0:
        #    util.export_pc(d_approx[0], args.save_path / f'exports/export_iter:{i}.xyz')
        #    util.export_pc(d1[0], args.save_path / f'targets/export_iter:{i}.xyz')
        #    util.export_pc(d2[0], args.save_path / f'sources/export_iter:{i}.xyz')
        #    torch.save(model.state_dict(), args.save_path / f'generators/model{i}.pt')


if __name__ == "__main__":
    print('Hola')
    parser = options.get_parser('Train Self-Sampling generator')
    args = options.parse_args(parser)
    train(args)