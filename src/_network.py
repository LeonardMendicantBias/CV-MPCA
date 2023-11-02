
class MyTransformer(nn.Module):

    def __init__(self,
        cameras: Dict[str, utils.Camera],
        num_classes: int,
        num_layers: int,
        num_levels: int,
        num_query: int,
        embed_dim: int,  # = 128
        pc_range,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cams = len(cameras)
        self.num_sbs = len(cameras)
        self.cameras = cameras

        self.query = nn.Embedding(num_query, embed_dim)
        self.to_anchor = nn.Linear(embed_dim, 3)

        self.decoder = Decoder(
            cameras=self.cameras,
            num_levels=num_levels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_layers=num_layers,
            pc_range=pc_range
        )

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.query.weight)
        
        nn.init.xavier_uniform_(self.to_anchor.weight)
        nn.init.zeros_(self.to_anchor.bias)

    def forward(self, visual_features):
        keys = list(visual_features.keys())
        keys_ = list(visual_features[keys[0]].keys())
        B = visual_features[keys[0]][keys_[0]].size(0)

        query = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        anchor = self.to_anchor(query).sigmoid()

        # decoder
        scores, positions, visibilities = self.decoder(
            query=query,
            key=None,
            value=visual_features,
            anchor=anchor
        )

        return scores, positions, visibilities

