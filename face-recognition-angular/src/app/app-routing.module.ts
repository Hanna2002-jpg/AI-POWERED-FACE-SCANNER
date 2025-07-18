import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ProjectStructureComponent } from './components/project-structure/project-structure.component';
import { DatasetManagerComponent } from './components/dataset-manager/dataset-manager.component';
import { TrainingPipelineComponent } from './components/training-pipeline/training-pipeline.component';
import { ModelEvaluationComponent } from './components/model-evaluation/model-evaluation.component';
import { ReportGeneratorComponent } from './components/report-generator/report-generator.component';

const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'structure', component: ProjectStructureComponent },
  { path: 'dataset', component: DatasetManagerComponent },
  { path: 'training', component: TrainingPipelineComponent },
  { path: 'evaluation', component: ModelEvaluationComponent },
  { path: 'report', component: ReportGeneratorComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }