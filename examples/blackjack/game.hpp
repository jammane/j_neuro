// Blackjack game rules shared by the trainer and the analysis tool (examples/blackjack).
#pragma once

#include <jneuro/jneuro.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace bj {

using namespace jneuro;
using D = std::uint32_t;

constexpr int kDecks = 5;          // a 5-deck shoe
constexpr int kRanks = 13;         // 2..10, J, Q, K, A
constexpr int kCardsPerHand = 7;
constexpr int kMaxHands = 4;       // limit on split hands per deal

// ------------------------------------------------------------------ cards
// rank index: 0..8 -> "2".."10", 9 J, 10 Q, 11 K, 12 A
inline bool is_ace(int r) { return r == 12; }
inline int points(int r) { return r <= 8 ? r + 2 : (r <= 11 ? 10 : 11); }
inline double input_value(int r) { return static_cast<double>(points(r)); }  // A = 11, J/Q/K = 10

inline int best_total(const std::vector<int>& cards, std::size_t n) {
    int total = 0, soft_aces = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total += points(cards[i]);
        if (is_ace(cards[i])) ++soft_aces;
    }
    while (total > 21 && soft_aces > 0) { total -= 10; --soft_aces; }  // ace counted as 1
    return total;
}

/// One deal: a freshly shuffled 5-deck shoe; cards are drawn from its top.
class Shoe {
public:
    explicit Shoe(std::uint64_t seed) : rng_(seed) {
        for (int r = 0; r < kRanks; ++r)
            for (int c = 0; c < 4 * kDecks; ++c) cards_.push_back(r);
    }
    int draw() {  // partial Fisher-Yates: only shuffles as far as cards are drawn
        const std::size_t j = pos_ + static_cast<std::size_t>(rng_.below(cards_.size() - pos_));
        std::swap(cards_[pos_], cards_[j]);
        return cards_[pos_++];
    }

private:
    Rng rng_;
    std::vector<int> cards_;
    std::size_t pos_ = 0;
};

struct Hand {
    std::vector<int> cards;  // 7 cards drawn up front
    std::size_t revealed = 2;
    bool doubled = false;
    bool done = false;
};

enum Action { Stand = 0, Hit = 1, Double = 2, Split = 3 };

inline Action decode(double y) {  // bound to 0..3, round up
    const double c = std::ceil(std::clamp(y, 0.0, 3.0));
    return static_cast<Action>(static_cast<int>(c));
}

/// One deal. Every card is drawn up front, in a fixed order: player, dealer, then one block
/// of 7 per possible split. Split k always uses block k, so every net that takes the same
/// splits sees the same cards. The dealer's play does not depend on the player's.
struct Deal {
    std::vector<int> player, dealer, split_cards;
    int dealer_total = 0;

    explicit Deal(std::uint64_t seed) {
        Shoe shoe(seed);
        for (int i = 0; i < kCardsPerHand; ++i) player.push_back(shoe.draw());
        for (int i = 0; i < kCardsPerHand; ++i) dealer.push_back(shoe.draw());
        for (int i = 0; i < (kMaxHands - 1) * kCardsPerHand; ++i) split_cards.push_back(shoe.draw());
        std::size_t dn = 2;  // dealer stands on 17 or more, hits on 16 or less
        while (best_total(dealer, dn) < 17 && dn < static_cast<std::size_t>(kCardsPerHand)) ++dn;
        dealer_total = best_total(dealer, dn);
    }
};

/// The player's side of a deal in progress; `h` is the hand being decided.
struct Table {
    std::vector<Hand> hands;
    std::size_t h = 0;
    std::size_t next_split_card = 0;

    explicit Table(const Deal& d) : hands(1) { hands[0].cards = d.player; }
    bool finished() const { return h >= hands.size(); }

    void act(const Deal& d, Action a) {
        Hand& hand = hands[h];
        switch (a) {
            case Stand:
                hand.done = true;
                break;
            case Hit:
                if (hand.revealed < kCardsPerHand) ++hand.revealed;
                if (hand.revealed == kCardsPerHand || best_total(hand.cards, hand.revealed) > 21) hand.done = true;
                break;
            case Double:
                if (hand.revealed < kCardsPerHand) ++hand.revealed;
                hand.doubled = true;
                hand.done = true;
                break;
            case Split: {
                const bool valid = hand.revealed == 2 && hand.cards[0] == hand.cards[1] &&
                                   hands.size() < static_cast<std::size_t>(kMaxHands);
                if (!valid) { hand.done = true; break; }  // invalid split: stand
                Hand fresh;
                fresh.cards.push_back(hand.cards[0]);      // first card starts the new hand
                hand.cards.erase(hand.cards.begin());
                hand.cards.push_back(d.split_cards[next_split_card++]);  // existing hand back to 7
                for (int i = 1; i < kCardsPerHand; ++i) fresh.cards.push_back(d.split_cards[next_split_card++]);
                hands.push_back(std::move(fresh));         // played after the current hand
                break;                                     // decide again on 2 revealed cards
            }
        }
        while (!finished() && hands[h].done) ++h;
    }

    /// Real blackjack payout, per hand: win +1, loss -1, push 0; a doubled hand pays +-2. A player
    /// bust loses even if the dealer busts too. A blackjack (the original, unsplit two cards
    /// totalling 21) and a 5-card charlie (five cards without busting; wins against any dealer
    /// hand) pay 1.5.
    double score(const Deal& d) const {
        double s = 0;
        for (const Hand& hand : hands) {
            const int t = best_total(hand.cards, hand.revealed);
            const double stake = hand.doubled ? 2 : 1;
            const bool natural = hands.size() == 1 && hand.revealed == 2 && t == 21 && !hand.doubled;
            const bool charlie = hand.revealed >= 5 && t <= 21;
            if (t > 21) s -= stake;                                             // player bust
            else if (natural || charlie) s += 1.5;                              // blackjack, 5-card charlie
            else if (d.dealer_total > 21 || t > d.dealer_total) s += stake;     // win
            else if (t < d.dealer_total) s -= stake;                            // loss
        }                                                                       // equal totals: push
        return s;
    }
};

/// Play one deal with `net`; returns the score. `observe(hand, deal, action)` sees every decision.
template <class Observe>
double play_deal(Net<D>& net, const Deal& deal, Observe&& observe) {
    Table table(deal);
    net.reset();  // s_neuron history covers one deal
    std::vector<double> in(9);
    while (!table.finished()) {
        const Hand& hand = table.hands[table.h];
        in[0] = input_value(deal.dealer[0]);
        in[1] = 0.0;  // the dealer's second card is the hole card: face down, not visible to the player
        for (int i = 0; i < kCardsPerHand; ++i)
            in[2 + i] = static_cast<std::size_t>(i) < hand.revealed ? input_value(hand.cards[i]) : 0.0;
        const Action a = decode(net.step(in)[0]);
        observe(hand, deal, a);
        table.act(deal, a);
    }
    return table.score(deal);
}
inline double play_deal(Net<D>& net, const Deal& deal) {
    return play_deal(net, deal, [](const Hand&, const Deal&, Action) {});
}

/// True if some sequence of decisions ends with a positive score (exhaustive search).
inline bool winnable(const Deal& deal, const Table& table) {
    if (table.finished()) return table.score(deal) > 0;
    for (Action a : {Stand, Hit, Double, Split}) {
        Table next = table;
        next.act(deal, a);
        if (winnable(deal, next)) return true;
    }
    return false;
}
inline bool winnable(const Deal& deal) { return winnable(deal, Table(deal)); }

/// The simple hit rule (a reference, not a net): hit below 12, or below 17 when the dealer's first
/// card is 7 or higher; otherwise stand. Never doubles or splits.
inline Action hit_rule(const Hand& hand, const Deal& deal) {
    const int total = best_total(hand.cards, hand.revealed), up = points(deal.dealer[0]);
    return (total < 12 || (total < 17 && up >= 7)) ? Hit : Stand;
}

/// The simple double rule: the hit rule, plus double down when the first two cards total 10 or 11
/// and the dealer's first card is lower than that total.
inline Action double_rule(const Hand& hand, const Deal& deal) {
    const int total = best_total(hand.cards, hand.revealed), up = points(deal.dealer[0]);
    if (hand.revealed == 2 && (total == 10 || total == 11) && up < total) return Double;
    return hit_rule(hand, deal);
}

/// Play one deal with a fixed rule; returns the score.
template <class Rule>
double play_rule(const Deal& deal, Rule rule) {
    Table table(deal);
    while (!table.finished()) table.act(deal, rule(table.hands[table.h], deal));
    return table.score(deal);
}
inline double play_hit_rule(const Deal& deal) { return play_rule(deal, hit_rule); }
inline double play_double_rule(const Deal& deal) { return play_rule(deal, double_rule); }

}  // namespace bj
