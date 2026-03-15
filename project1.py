import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import faiss
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run_nn_example():
    print("\nRunning simple neural network example")
    X,y = make_moons(n_samples=800,noise=0.2)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    X_train = torch.tensor(X_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.long)
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(100):
        logits = model(X_train.to(device))
        loss = loss_fn(logits,y_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print("epoch:",epoch,"loss:",loss.item())
    print("neural network training finished")

class LSTMModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1,16)
        self.lstm = nn.LSTM(16,32,batch_first=True)
        self.fc = nn.Linear(32,2)
    def forward(self,x):
        x = self.embedding(x)
        _,(hidden,_) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def run_lstm():
    print("\nRunning LSTM example")
    texts = [
        "i love this movie",
        "this movie is great",
        "really amazing acting",
        "i hate this movie",
        "worst film ever",
        "bad acting"
    ]
    labels = [1,1,1,0,0,0]
    words = " ".join(texts).split()
    vocab = {w:i+1 for i,w in enumerate(set(words))}
    def encode(sentence):
        return [vocab[w] for w in sentence.split()]
    encoded = [encode(t) for t in texts]
    max_len = max(len(x) for x in encoded)
    padded = [x + [0]*(max_len-len(x)) for x in encoded]
    X = torch.tensor(padded)
    y = torch.tensor(labels)
    model = LSTMModel(len(vocab))
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    for epoch in range(80):
        logits = model(X)
        loss = F.cross_entropy(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print("epoch:",epoch,"loss:",loss.item())
    print("LSTM example done")

class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim,embed_dim)
        self.k = nn.Linear(embed_dim,embed_dim)
        self.v = nn.Linear(embed_dim,embed_dim)
    def forward(self,x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = Q @ K.transpose(-2,-1)
        scores = scores / np.sqrt(x.size(-1))
        weights = torch.softmax(scores,dim=-1)
        out = weights @ V
        return out

def attention_demo():
    print("\nRunning self attention demo")
    x = torch.randn(1,5,16)
    attn = SelfAttention(16)
    out = attn(x)
    print("input shape:",x.shape)
    print("output shape:",out.shape)

def rag_demo():
    print("\nRunning simple RAG retrieval demo")
    documents = [
        "Artificial intelligence is transforming industries",
        "Deep learning uses neural networks",
        "Transformers power GPT models",
        "RAG combines retrieval with generation",
        "Vector databases help semantic search"
    ]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(documents)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    print("documents indexed:",index.ntotal)
    query = "how GPT models work"
    q_embed = embedder.encode([query])
    D,I = index.search(np.array(q_embed),k=2)
    print("\nquery:",query)
    print("top matches:")
    for i in I[0]:
        print("-",documents[i])

print("\nStarting project\n")
run_nn_example()
run_lstm()
attention_demo()
rag_demo()
print("\nProject finished")