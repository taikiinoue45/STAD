import efficientnet_pytorch
from torch.nn import functional as F


class EfficientNet(efficientnet_pytorch.EfficientNet):
    def get_feature_list(self, xs):
        features = []

        xs = self._swish(self._bn0(self._conv_stem(xs)))
        features.append(F.adaptive_avg_pool2d(xs, 1))

        xs_prev = xs
        for i, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self._blocks)
            xs = block(xs, drop_connect_rate=drop_connect_rate)

            if (xs_prev.shape[1] != xs.shape[1] and i != 0) or i == len(self._blocks) - 1:
                features.append(F.adaptive_avg_pool2d(xs_prev, 1))

            xs_prev = xs

        xs = self._swish(self._bn1(self._conv_head(xs)))
        features.append(F.adaptive_avg_pool2d(xs, 1))

        return features
