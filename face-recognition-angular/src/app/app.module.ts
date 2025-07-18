import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

// Angular Material
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';

// Chart.js
import { NgChartsModule } from 'ng2-charts';

// Lucide Icons
import { LucideAngularModule, Home, Database, Brain, Target, FileText, FolderTree, Settings, Play, Pause, Download, Eye, Upload } from 'lucide-angular';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ProjectStructureComponent } from './components/project-structure/project-structure.component';
import { DatasetManagerComponent } from './components/dataset-manager/dataset-manager.component';
import { TrainingPipelineComponent } from './components/training-pipeline/training-pipeline.component';
import { ModelEvaluationComponent } from './components/model-evaluation/model-evaluation.component';
import { ReportGeneratorComponent } from './components/report-generator/report-generator.component';
import { NavigationComponent } from './components/navigation/navigation.component';
import { MetricCardComponent } from './components/metric-card/metric-card.component';
import { PerformanceChartComponent } from './components/performance-chart/performance-chart.component';

// Services
import { DataService } from './services/data.service';
import { TrainingService } from './services/training.service';
import { EvaluationService } from './services/evaluation.service';
import { ReportService } from './services/report.service';
import { PDFService } from './services/pdf.service';

@NgModule({
  declarations: [
    AppComponent,
    DashboardComponent,
    ProjectStructureComponent,
    DatasetManagerComponent,
    TrainingPipelineComponent,
    ModelEvaluationComponent,
    ReportGeneratorComponent,
    NavigationComponent,
    MetricCardComponent,
    PerformanceChartComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    
    // Angular Material
    MatToolbarModule,
    MatButtonModule,
    MatCardModule,
    MatTabsModule,
    MatProgressBarModule,
    MatTableModule,
    MatIconModule,
    MatSnackBarModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatCheckboxModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    
    // Chart.js
    NgChartsModule,
    
    // Lucide Icons
    LucideAngularModule.pick({ Home, Database, Brain, Target, FileText, FolderTree, Settings, Play, Pause, Download, Eye, Upload })
  ],
  providers: [
    DataService,
    TrainingService,
    EvaluationService,
    ReportService,
    PDFService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }