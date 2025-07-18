import { Component } from '@angular/core';
import { Router } from '@angular/router';

interface NavItem {
  id: string;
  label: string;
  icon: string;
  route: string;
}

@Component({
  selector: 'app-navigation',
  template: `
    <mat-toolbar class="navigation-toolbar">
      <div class="nav-brand">
        <lucide-icon name="brain" class="brand-icon"></lucide-icon>
        <span class="brand-text">Face Scan Master</span>
      </div>
      
      <div class="nav-items">
        <button 
          mat-button 
          *ngFor="let item of navItems"
          [class.active]="isActive(item.route)"
          (click)="navigate(item.route)"
          class="nav-button">
          <lucide-icon [name]="item.icon" class="nav-icon"></lucide-icon>
          <span class="nav-label">{{ item.label }}</span>
        </button>
      </div>
    </mat-toolbar>
  `,
  styles: [`
    .navigation-toolbar {
      background: rgba(30, 41, 59, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid rgba(71, 85, 105, 0.5);
      color: white;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 2rem;
    }
    
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .brand-icon {
      width: 2rem;
      height: 2rem;
      color: #60a5fa;
    }
    
    .brand-text {
      font-size: 1.25rem;
      font-weight: 700;
    }
    
    .nav-items {
      display: flex;
      gap: 0.25rem;
    }
    
    .nav-button {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      color: #cbd5e1;
      transition: all 0.2s;
    }
    
    .nav-button:hover {
      background: rgba(71, 85, 105, 0.5);
      color: white;
    }
    
    .nav-button.active {
      background: #2563eb;
      color: white;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .nav-icon {
      width: 1rem;
      height: 1rem;
    }
    
    .nav-label {
      font-size: 0.875rem;
    }
    
    @media (max-width: 768px) {
      .nav-label {
        display: none;
      }
    }
  `]
})
export class NavigationComponent {
  navItems: NavItem[] = [
    { id: 'dashboard', label: 'Dashboard', icon: 'home', route: '/dashboard' },
    { id: 'structure', label: 'Scan Structure', icon: 'folder-tree', route: '/structure' },
    { id: 'dataset', label: 'Scan Dataset', icon: 'database', route: '/dataset' },
    { id: 'training', label: 'Scan Training', icon: 'brain', route: '/training' },
    { id: 'evaluation', label: 'Scan Evaluation', icon: 'target', route: '/evaluation' },
    { id: 'report', label: 'Scan Reports', icon: 'file-text', route: '/report' }
  ];

  constructor(private router: Router) {}

  navigate(route: string): void {
    this.router.navigate([route]);
  }

  isActive(route: string): boolean {
    return this.router.url === route;
  }
}