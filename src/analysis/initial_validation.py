#!/usr/bin/env python3
"""
COVID-19 Infection Simulator Validator for Barnsdall, OK
Quick validation script to compare simulation results against real-world data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class InfectionValidator:
    def __init__(self, csv_file="../../data/raw/run1/infection_chains.csv"):
        """Initialize validator with simulation data"""
        self.csv_file = csv_file
        self.simulation_data = None
        self.real_world_benchmarks = {
            'barnsdall_population': 1180,  # Approximate population of Barnsdall, OK
            'known_cases_nov_2020': 9,     # Confirmed cases Nov 9, 2020
            'peak_infection_rate': 0.05,   # Estimated 5% peak infection rate for small towns
            'r0_delta': 5.0,               # Delta variant R0
            'r0_omicron': 9.5,             # Omicron variant R0
            'avg_generation_time': 5.2,    # Days between infections
        }
        
    def load_simulation_data(self):
        """Load and process simulation data"""
        try:
            self.simulation_data = pd.read_csv(self.csv_file)
            print(f"✓ Loaded {len(self.simulation_data)} infection records")
            print(f"Columns: {list(self.simulation_data.columns)}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: Could not find {self.csv_file}")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def analyze_simulation_metrics(self):
        """Analyze key metrics from simulation"""
        if self.simulation_data is None:
            print("No simulation data loaded")
            return {}
        
        df = self.simulation_data
        
        metrics = {
            'total_infections': len(df),
            'unique_people': len(df['infected_person_id'].unique()),
            'variant_distribution': df['variant'].value_counts().to_dict(),
            'infection_timeline': {
                'min_timestep': df['timestep'].min(),
                'max_timestep': df['timestep'].max(),
                'duration_days': (df['timestep'].max() - df['timestep'].min()) / 24,  # Assuming hourly timesteps
            },
            'location_spread': len(df['location_id'].unique()),
            'avg_parameters': {
                'mask_compliance': df['mask'].mean(),
                'vaccine_rate': df['vaccine'].mean(),
                'capacity_limit': df['capacity'].mean(),
                'lockdown_effectiveness': df['lockdown'].mean(),
                'self_isolation_rate': df['selfiso'].mean(),
            }
        }
        
        return metrics
    
    def calculate_reproduction_number(self):
        """Estimate effective reproduction number from simulation"""
        if self.simulation_data is None:
            return None
            
        df = self.simulation_data
        
        # Count secondary infections per infector
        infector_counts = df.groupby('infector_id').size()
        r_effective = infector_counts.mean()
        
        # By variant
        variant_r = {}
        for variant in df['variant'].unique():
            variant_data = df[df['variant'] == variant]
            variant_infector_counts = variant_data.groupby('infector_id').size()
            variant_r[variant] = variant_infector_counts.mean()
        
        return {
            'overall_r_effective': r_effective,
            'by_variant': variant_r
        }
    
    def validate_against_benchmarks(self, metrics):
        """Compare simulation metrics against real-world benchmarks"""
        validation_results = {}
        
        # Population infection rate
        infection_rate = metrics['unique_people'] / self.real_world_benchmarks['barnsdall_population']
        validation_results['infection_rate'] = {
            'simulated': infection_rate,
            'benchmark': self.real_world_benchmarks['peak_infection_rate'],
            'valid': abs(infection_rate - self.real_world_benchmarks['peak_infection_rate']) < 0.03,
            'comment': f"{'✓' if abs(infection_rate - self.real_world_benchmarks['peak_infection_rate']) < 0.03 else '✗'} Infection rate within reasonable range"
        }
        
        # R number validation
        r_data = self.calculate_reproduction_number()
        if r_data:
            for variant, r_val in r_data['by_variant'].items():
                benchmark_r = self.real_world_benchmarks.get(f'r0_{variant.lower()}', 3.0)
                validation_results[f'r0_{variant}'] = {
                    'simulated': r_val,
                    'benchmark': benchmark_r,
                    'valid': abs(r_val - benchmark_r) < 2.0,
                    'comment': f"{'✓' if abs(r_val - benchmark_r) < 2.0 else '✗'} {variant} R-value reasonably close to real-world"
                }
        
        # Timeline validation (50 days expected)
        timeline_days = metrics['infection_timeline']['duration_days']
        validation_results['timeline'] = {
            'simulated': timeline_days,
            'benchmark': 50,
            'valid': abs(timeline_days - 50) < 10,
            'comment': f"{'✓' if abs(timeline_days - 50) < 10 else '✗'} Timeline matches expected 50-day period"
        }
        
        return validation_results
    
    def generate_visualizations(self, metrics):
        """Generate validation visualizations"""
        if self.simulation_data is None:
            return
            
        df = self.simulation_data
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('COVID-19 Simulation Validation: Barnsdall, OK', fontsize=16, fontweight='bold')
        
        # 1. Infections over time
        df['day'] = df['timestep'] // 24
        daily_infections = df.groupby('day').size()
        axes[0,0].plot(daily_infections.index, daily_infections.values, 'b-', linewidth=2)
        axes[0,0].set_title('Daily New Infections')
        axes[0,0].set_xlabel('Day')
        axes[0,0].set_ylabel('New Infections')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Variant distribution
        variant_counts = df['variant'].value_counts()
        axes[0,1].pie(variant_counts.values, labels=variant_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Variant Distribution')
        
        # 3. Location spread
        location_infections = df.groupby('location_id').size().sort_values(ascending=False)[:10]
        axes[0,2].bar(range(len(location_infections)), location_infections.values)
        axes[0,2].set_title('Top 10 Infection Locations')
        axes[0,2].set_xlabel('Location ID (ranked)')
        axes[0,2].set_ylabel('Infections')
        
        # 4. R-number distribution
        infector_counts = df.groupby('infector_id').size()
        axes[1,0].hist(infector_counts.values, bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(infector_counts.mean(), color='red', linestyle='--', 
                         label=f'Mean R = {infector_counts.mean():.2f}')
        axes[1,0].set_title('Secondary Infections Distribution')
        axes[1,0].set_xlabel('Secondary Infections per Person')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 5. Parameter effectiveness
        params = ['mask', 'vaccine', 'capacity', 'lockdown', 'selfiso']
        param_values = [df[param].iloc[0] for param in params]
        axes[1,1].bar(params, param_values, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[1,1].set_title('Intervention Parameters')
        axes[1,1].set_ylabel('Effectiveness (0-1)')
        axes[1,1].set_ylim(0, 1)
        plt.setp(axes[1,1].get_xticklabels(), rotation=45)
        
        # 6. Cumulative infections
        cumulative_infections = daily_infections.cumsum()
        axes[1,2].plot(cumulative_infections.index, cumulative_infections.values, 'g-', linewidth=2)
        axes[1,2].axhline(y=self.real_world_benchmarks['known_cases_nov_2020'], 
                         color='red', linestyle='--', label='Known cases (Nov 2020)')
        axes[1,2].set_title('Cumulative Infections')
        axes[1,2].set_xlabel('Day')
        axes[1,2].set_ylabel('Total Cases')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('infection_validation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_validation(self):
        """Run complete validation process"""
        print("🦠 COVID-19 Infection Simulator Validator")
        print("=" * 50)
        
        # Load data
        if not self.load_simulation_data():
            return
        
        # Analyze metrics
        print("\n📊 Analyzing simulation metrics...")
        metrics = self.analyze_simulation_metrics()
        
        print(f"\n📈 SIMULATION SUMMARY:")
        print(f"Total infections: {metrics['total_infections']}")
        print(f"Unique people infected: {metrics['unique_people']}")
        print(f"Population infection rate: {metrics['unique_people']/self.real_world_benchmarks['barnsdall_population']*100:.1f}%")
        print(f"Simulation duration: {metrics['infection_timeline']['duration_days']:.1f} days")
        print(f"Locations involved: {metrics['location_spread']}")
        
        print(f"\n🧬 VARIANT BREAKDOWN:")
        for variant, count in metrics['variant_distribution'].items():
            print(f"  {variant}: {count} cases ({count/metrics['total_infections']*100:.1f}%)")
        
        print(f"\n🛡️ INTERVENTION PARAMETERS:")
        for param, value in metrics['avg_parameters'].items():
            print(f"  {param.replace('_', ' ').title()}: {value:.3f}")
        
        # Calculate R-number
        print(f"\n🔬 REPRODUCTION NUMBER ANALYSIS:")
        r_data = self.calculate_reproduction_number()
        if r_data:
            print(f"Overall R-effective: {r_data['overall_r_effective']:.2f}")
            for variant, r_val in r_data['by_variant'].items():
                benchmark = self.real_world_benchmarks.get(f'r0_{variant.lower()}', 'Unknown')
                print(f"  {variant}: {r_val:.2f} (benchmark: {benchmark})")
        
        # Validation
        print(f"\n✅ VALIDATION RESULTS:")
        validation = self.validate_against_benchmarks(metrics)
        
        all_valid = True
        for metric, result in validation.items():
            print(f"  {result['comment']}")
            if not result['valid']:
                all_valid = False
        
        print(f"\n🎯 OVERALL VALIDATION: {'PASS' if all_valid else 'NEEDS ADJUSTMENT'}")
        
        if not all_valid:
            print("\n💡 RECOMMENDATIONS:")
            for metric, result in validation.items():
                if not result['valid']:
                    if 'infection_rate' in metric:
                        print(f"  - Consider adjusting transmission parameters to match ~5% infection rate")
                    elif 'r0' in metric:
                        print(f"  - Review {metric} - may need parameter tuning for realistic transmission")
                    elif 'timeline' in metric:
                        print(f"  - Timeline seems off - check timestep calculation")
        
        # Generate plots
        print(f"\n📊 Generating validation plots...")
        self.generate_visualizations(metrics)
        print(f"Plots saved as 'infection_validation_plots.png'")
        
        return metrics, validation

# Main execution
if __name__ == "__main__":
    validator = InfectionValidator("infection_chains.csv")
    metrics, validation = validator.run_validation()