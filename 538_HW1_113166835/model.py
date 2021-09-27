"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.records import array
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start
        # import pdb; pdb.set_trace()
        
        center_embeds = self.center_embeddings(center_word)
        context_embeds = self.context_embeddings(context_word)
        
        MatrixMultiplication = torch.matmul(center_embeds ,  context_embeds.T)
        totalSum=torch.sum(torch.exp(MatrixMultiplication), dim=1)
        logofsum=torch.log(totalSum)
        
        MatrixMultiplication1=(torch.multiply(center_embeds, context_embeds))
        totalsum1=torch.sum(MatrixMultiplication1, dim=1)


        # # torch.exp()

        loss=torch.mean(logofsum-totalsum1)
        ### TODO(students): end
        # loss=0
        return loss
    
    def negative_sampling(self, center_word, context_word):
        ### TODO(students): start
        
        center_embeds = self.center_embeddings(center_word)
        context_embeds = self.context_embeddings(context_word)
        
        batch_size=center_word.size()[0]
        probability=(self.counts)/sum(self.counts)
        
        k=5
        neg= np.random.choice(len(self.counts),(batch_size,k), replace=False, p=probability)
        Negative_embeds=self.context_embeddings.weight[neg]
        
        sum1 = torch.log(torch.sigmoid(torch.sum(torch.multiply(center_embeds , context_embeds),dim=1)))
        center_embeds = center_embeds.reshape((center_embeds.shape[0], center_embeds.shape[1], 1))
        
        sum2 = torch.sum(torch.log(torch.sigmoid(torch.sum(-torch.matmul(Negative_embeds, center_embeds),dim=2))),dim=1)
        loss=torch.mean(-sum2-sum1)
        ### TODO(students): end

        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            