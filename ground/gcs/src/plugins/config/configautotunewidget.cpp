
#include "configautotunewidget.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <QList>
#include "relaytuningsettings.h"
#include "relaytuning.h"
#include "stabilizationsettings.h"
#include "modulesettings.h"

ConfigAutotuneWidget::ConfigAutotuneWidget(QWidget *parent) :
    ConfigTaskWidget(parent)
{
    m_autotune = new Ui_AutotuneWidget();
    m_autotune->setupUi(this);

    // Connect automatic signals
    autoLoadWidgets();
    disableMouseWheelEvents();

    // Whenever any value changes compute new potential stabilization settings
    connect(m_autotune->rateTuning, SIGNAL(valueChanged(int)), this, SLOT(recomputeStabilization()));

    addUAVObject("ModuleSettings");
    addWidget(m_autotune->enableAutoTune);

    RelayTuning *relayTuning = RelayTuning::GetInstance(getObjectManager());
    Q_ASSERT(relayTuning);
    if(relayTuning)
        connect(relayTuning, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(recomputeStabilization()));

    // Connect the apply button for the stabilization settings
    connect(m_autotune->useComputedValues, SIGNAL(pressed()), this, SLOT(saveStabilization()));
}

/**
  * Apply the stabilization settings computed
  */
void ConfigAutotuneWidget::saveStabilization()
{
    StabilizationSettings *stabilizationSettings = StabilizationSettings::GetInstance(getObjectManager());
    Q_ASSERT(stabilizationSettings);
    if(!stabilizationSettings)
        return;

    // Make sure to recompute in case the other stab settings changed since
    // the last time
    recomputeStabilization();

    // Apply this data to the board
    stabilizationSettings->setData(stabSettings);
    stabilizationSettings->updated();
}

/**
  * Called whenever the gain ratios or measured values
  * are changed
  */
void ConfigAutotuneWidget::recomputeStabilization()
{
    RelayTuning *relayTuning = RelayTuning::GetInstance(getObjectManager());
    Q_ASSERT(relayTuning);
    if(!relayTuning)
        return;

    StabilizationSettings *stabilizationSettings = StabilizationSettings::GetInstance(getObjectManager());
    Q_ASSERT(stabilizationSettings);
    if(!stabilizationSettings)
        return;

    RelayTuning::DataFields relayTuningData = relayTuning->getData();
    stabSettings = stabilizationSettings->getData();

    // These three parameters define the desired response properties
    // - rate scale in the fraction of the natural speed of the system
    //   to strive for.
    // - damp is the amount of damping in the system. higher values
    //   make oscillations less likely
    // - ghf is the amount of high frequency gain and limits the influence
    //   of noise
    const double scale = m_autotune->rateTuning->value() / 100.0;
    const double damp = m_autotune->rateDamp->value() / 100.0;
    const double ghf = 0.02;

    double tau = exp(relayTuningData.Tau);
    double beta_roll = relayTuningData.Beta[RelayTuning::BETA_ROLL];
    double beta_pitch = relayTuningData.Beta[RelayTuning::BETA_PITCH];

    double wn = 1/tau;
    double tau_d = 0;
    for (int i = 0; i < 30; i++) {
        double tau_d_roll = (2*damp*tau*wn - 1)/(4*tau*damp*damp*wn*wn - 2*damp*wn - tau*wn*wn + exp(beta_roll)*ghf);
        double tau_d_pitch = (2*damp*tau*wn - 1)/(4*tau*damp*damp*wn*wn - 2*damp*wn - tau*wn*wn + exp(beta_pitch)*ghf);

        // Select the slowest filter property
        tau_d = (tau_d_roll > tau_d_pitch) ? tau_d_roll : tau_d_pitch;
        wn = (tau + tau_d) / (tau*tau_d) / (2 * damp + 2);
    }



    qDebug() << "wn: " << wn;
    qDebug() << "tau_d: " << tau_d;

    // Set the real pole position
    const double a = ((tau+tau_d) / tau / tau_d - 2 * damp * wn) / 2;
    const double b = ((tau+tau_d) / tau / tau_d - 2 * damp * wn - a);

    qDebug() << "a: " << a << " b: " << b;

    // For now just run over roll and pitch
    for (int i = 0; i < 2; i++) {
        double beta = exp(relayTuningData.Beta[i]);

        double ki = a * b * wn * wn * tau * tau_d / beta;
        double kp = tau * tau_d * ((a+b)*wn*wn + 2*a*b*damp*wn) / beta - ki*tau_d;
        double kd = (tau * tau_d * (a*b + wn*wn + (a+b)*2*damp*wn) - 1) / beta - kp * tau_d;

        switch(i) {
        case 0: // Roll

            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KP] = kp;
            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KI] = ki;
            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KD] = kd;
            break;
        case 1: // Pitch
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KP] = kp;
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KI] = ki;
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KD] = kd;
            break;
        }
    }
    stabSettings.DerivativeCutoff = 1 / (2*M_PI*tau_d);

    // Display these computed settings
    m_autotune->rollRateKp->setText(QString::number(stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KP]));
    m_autotune->rollRateKi->setText(QString::number(stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KI]));
    m_autotune->rollRateKd->setText(QString::number(stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KD]));
    m_autotune->pitchRateKp->setText(QString::number(stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KP]));
    m_autotune->pitchRateKi->setText(QString::number(stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KI]));
    m_autotune->pitchRateKd->setText(QString::number(stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KD]));

    m_autotune->derivativeCutoff->setText(QString::number(stabSettings.DerivativeCutoff));
    m_autotune->rollTau->setText(QString::number(tau,'g',3));
    m_autotune->pitchTau->setText(QString::number(tau,'g',3));
    m_autotune->wn->setText(QString::number(wn / 2 / M_PI, 'f', 1));
}

void ConfigAutotuneWidget::refreshWidgetsValues(UAVObject *obj)
{
    ModuleSettings *moduleSettings = ModuleSettings::GetInstance(getObjectManager());
    if(obj==moduleSettings)
    {
        bool dirtyBack=isDirty();
        ModuleSettings::DataFields moduleSettingsData = moduleSettings->getData();
        m_autotune->enableAutoTune->setChecked(
            moduleSettingsData.AdminState[ModuleSettings::ADMINSTATE_AUTOTUNE] == ModuleSettings::ADMINSTATE_ENABLED);
        setDirty(dirtyBack);
    }
    ConfigTaskWidget::refreshWidgetsValues(obj);
}
void ConfigAutotuneWidget::updateObjectsFromWidgets()
{
    ModuleSettings *moduleSettings = ModuleSettings::GetInstance(getObjectManager());
    ModuleSettings::DataFields moduleSettingsData = moduleSettings->getData();
    moduleSettingsData.AdminState[ModuleSettings::ADMINSTATE_AUTOTUNE] =
         m_autotune->enableAutoTune->isChecked() ? ModuleSettings::ADMINSTATE_ENABLED : ModuleSettings::ADMINSTATE_DISABLED;
    moduleSettings->setData(moduleSettingsData);
    ConfigTaskWidget::updateObjectsFromWidgets();
}
