class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.device = 'cuda'
        C.num_workers = 4
        C.max_iters = None
        C.batch_size = 128
        C.max_length = 512
        C.learning_rate = 8e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset,val_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)
        self.device ='cuda'
        self.model = self.model.to(self.device)
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config)
        batch_size = self.config.batch_size
        def get_batch(mode):
            batch_size = 64
            if mode == "train":
              data = train_dataset
              pos_train = torch.tensor(np.array([data[i:i+1,:] for i in range(data.shape[0]) if data[i,-1].item() == 1]).reshape(len([data[i:i+1,:] for i in range(data.shape[0]) if data[i,-1].item() == 1]),513))
              neg_train = torch.tensor(np.array([data[i:i+1,:] for i in range(data.shape[0]) if data[i,-1].item() == 0]).reshape(len([data[i:i+1,:] for i in range(data.shape[0]) if data[i,-1].item() == 0]),513))
              N = np.random.randint(batch_size)
              ix_pos = np.random.randint(pos_train.shape[0]-N)
              ix_neg = np.random.randint(neg_train.shape[0]-batch_size-N)
              pos_data = pos_train[ix_pos:ix_pos+N,:]
              neg_data = neg_train[ix_neg:ix_neg+batch_size-N,:]
              data = torch.concat((pos_data,neg_data))
              seq = data[:,:-1]
              targets = torch.zeros((batch_size,1), device="cuda")
              o = -1
              for i in data:
                o+=1
                if i[-1].item() == 1:
                  targets[o] = 1
                #else:
                  #targets[o][1] = 1
            else:
              data = val_dataset
              ix = np.random.randint(data.shape[0]-batch_size)
              data = data[ix:ix+batch_size,:]
              seq = data[:,:-1]
              targets = torch.zeros((batch_size,1), device="cuda")
              o = -1
              for i in data:
                o+=1
                if i[-1].item() == 1:
                  targets[o] = 1
                #else:
                 # targets[o][1] = 1
            targets = targets.view(batch_size,1).to("cuda")
            seq = seq.view(batch_size,512).to("cuda")
            return seq, targets

        @torch.no_grad
        def cross_val():
          model.eval()
          out = []
          for i in ["train","val"]:
            losses = torch.zeros(200+1)
            for k in range(200):
              X,Y = get_batch(i)
              logits,loss = model(X,Y)
              losses[k]=loss.item()
              out1 = losses.mean()
            out.append(out1)
          model.train()
          return out
        losses = cross_val()
        LOSS = [losses]
        print("\n[train loss = {k}, val loss =  {j}]\n".format(k = losses[0],j = losses[1]))
        model.train()
        for epoch in range(20):
          print("[epoch {o}] \n".format(o = epoch))
          iters = 200
          for i in range(iters):
            if i == 0:
              Y = "| ="
            elif i == iters -1 :
              Y = "=> 100% |"
            else:
              if i%(int(iters/50)) == 0 :
                Y = "="
              else:
                Y = ""
            print("{y}".format(y = Y),end="")
            x, y = get_batch("train")
            logits, self.loss = model(x, y)
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
          losses = cross_val()
          LOSS.append(losses)
          print(LOSS)
          print("\n[train loss = {k}, val loss =  {j}]\n".format(k = losses[0],j = losses[1]))
        PATH = "model {h}".format(h = model_config.model_type)
        torch.save(model.state_dict(), PATH)
