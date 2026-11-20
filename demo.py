"""Run the bundled example, or query another Boolean XMLBIF network."""

import argparse
from pathlib import Path

from BNReasoner import BNReasoner


def assignment(text):
    try:
        name, value = text.rsplit('=', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use VARIABLE=true or VARIABLE=false') from exc
    if not name or value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Use VARIABLE=true or VARIABLE=false')
    return name, value.lower() == 'true'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, default=Path(__file__).parent / 'testing/dog_problem.BIFXML')
    parser.add_argument('--query', nargs='+', help='Query variable names; quote names containing spaces')
    parser.add_argument('--evidence', nargs='*', type=assignment, default=[])
    parser.add_argument('--order', choices=['min-fill', 'min-degree'], default='min-fill')
    args = parser.parse_args()
    if len(dict(args.evidence)) != len(args.evidence):
        parser.error('Each evidence variable may be given only once')
    try:
        reasoner = BNReasoner(args.network)
        if args.query is not None:
            result = reasoner.query(args.query, dict(args.evidence), args.order)
            print(result.to_string(index=False, float_format=lambda p: f'{p:.6f}'))
            return
        if args.evidence or args.network != Path(__file__).parent / 'testing/dog_problem.BIFXML':
            parser.error('Provide --query when using a custom network or evidence')
        prior = reasoner.query(['family-out'])
        posterior = reasoner.query(['family-out'], {'hear-bark': True})
        p_prior = float(prior.loc[prior['family-out'], 'p'].iloc[0])
        p_posterior = float(posterior.loc[posterior['family-out'], 'p'].iloc[0])
        print('Dog-problem teaching example')
        print(f'P(family out)                 = {p_prior:.6f}')
        print(f'P(family out | hear barking)  = {p_posterior:.6f}')
        print('Hearing barking changes the probability; it does not establish the cause.')
        print('\nPosterior distribution:')
        print(posterior.to_string(index=False, float_format=lambda p: f'{p:.6f}'))
        print('\nMost likely family-out assignment given barking:')
        print(reasoner.map(['family-out'], {'hear-bark': True}))
        print('\nAre light-on and hear-bark independent given family-out?')
        print(reasoner.d_separated({'light-on'}, {'family-out'}, {'hear-bark'}))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
