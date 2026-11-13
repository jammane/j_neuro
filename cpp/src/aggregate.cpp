#include <jneuro/aggregate.hpp>
#include <jneuro/numerics.hpp>

#include <algorithm>

namespace jneuro {

double aggregate(AggOp op, const std::vector<double>& values, const std::vector<bool>& present) {
    std::vector<double> scratch;
    return aggregate(op, values, present, scratch);
}

double aggregate(AggOp op, const std::vector<double>& values, const std::vector<bool>& present,
                 std::vector<double>& scratch) {
    switch (op) {
        case AggOp::Mean:
            return compensated_mean(values);
        case AggOp::Median: {
            scratch.assign(values.begin(), values.end());
            std::sort(scratch.begin(), scratch.end());
            return scratch[scratch.size() / 2];
        }
        case AggOp::Min: {
            bool any = false;
            double m = 0.0;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (!present[i]) continue;
                if (!any || values[i] < m) m = values[i];
                any = true;
            }
            return any ? m : 0.0;
        }
        case AggOp::Lag:  // s_neurons only: the oldest history value (spec/04)
            return values[0];
        case AggOp::Max: {
            double m = values[0];
            for (std::size_t i = 1; i < values.size(); ++i)
                if (values[i] > m) m = values[i];
            return m;
        }
    }
    return 0.0;
}

}  // namespace jneuro
