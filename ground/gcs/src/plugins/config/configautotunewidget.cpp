
#include "configautotunewidget.h"

#include <QDebug>
#include <QStringList>
#include <QWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>
#include <QList>
#include "systemident.h"
#include "stabilizationsettings.h"
#include "modulesettings.h"
#include "coreplugin/generalsettings.h"
#include "utils/phpbb.h"
#include "extensionsystem/pluginmanager.h"
#include <QMessageBox>

#define FORUM_SHARING_FORUM     24
#define FORUM_SHARING_THREAD    255

ConfigAutotuneWidget::ConfigAutotuneWidget(QWidget *parent) :
    ConfigTaskWidget(parent)
{
    m_autotune = new Ui_AutotuneWidget();
    m_autotune->setupUi(this);

    // Connect automatic signals
    autoLoadWidgets();
    disableMouseWheelEvents();

    // Whenever any value changes compute new potential stabilization settings
    connect(m_autotune->rateDamp, SIGNAL(valueChanged(int)), this, SLOT(recomputeStabilization()));
    connect(m_autotune->rateNoise, SIGNAL(valueChanged(int)), this, SLOT(recomputeStabilization()));

    addUAVObject("ModuleSettings");
    addWidget(m_autotune->enableAutoTune);

    SystemIdent *systemIdent = SystemIdent::GetInstance(getObjectManager());
    Q_ASSERT(systemIdent);
    if(systemIdent)
        connect(systemIdent, SIGNAL(objectUpdated(UAVObject*)), this, SLOT(recomputeStabilization()));

    // Connect the apply button for the stabilization settings
    connect(m_autotune->useComputedValues, SIGNAL(pressed()), this, SLOT(saveStabilization()));

    connect(m_autotune->shareDataPB, SIGNAL(pressed()),this, SLOT(onShareData()));
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

    // Check the settings are reasonable, or if not have the
    // user confirm they want to continue.
    SystemIdent *systemIdent = SystemIdent::GetInstance(getObjectManager());
    Q_ASSERT(systemIdent);
    if(!systemIdent)
        return;
    if (approveSettings(systemIdent->getData()) == false)
        return;

    // Make sure to recompute in case the other stab settings changed since
    // the last time
    recomputeStabilization();

    // Apply this data to the board
    stabilizationSettings->setData(stabSettings);
    stabilizationSettings->updated();
}

void ConfigAutotuneWidget::onShareData()
{
    forumInteractionForm = new Utils::ForumInteractionForm(this);
    connect(forumInteractionForm, SIGNAL(finished(int)), this, SLOT(onForumInteractionSet(int)));
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if (!settings->getForumPassword().isEmpty()) {
       forumInteractionForm->setPassword(settings->getForumPassword());
       forumInteractionForm->setUserName(settings->getForumUser());
    }
    forumInteractionForm->setObservations(settings->getObservations());
    forumInteractionForm->setAircraftDescription(settings->getAircraftDescription());
    forumInteractionForm->show();
    forumInteractionForm->raise();
    forumInteractionForm->activateWindow();
}

void ConfigAutotuneWidget::onForumInteractionSet(int value)
{
    if (!value) {
        forumInteractionForm->deleteLater();
        return;
    }
    ExtensionSystem::PluginManager *pm=ExtensionSystem::PluginManager::instance();
    Core::Internal::GeneralSettings * settings=pm->getObject<Core::Internal::GeneralSettings>();
    if (forumInteractionForm->getSaveCredentials()) {
        settings->setForumPassword(forumInteractionForm->getPassword());
        settings->setForumUser(forumInteractionForm->getUserName());
    } else {
        settings->setForumPassword("");
        settings->setForumUser("");
    }
    settings->setObservations(forumInteractionForm->getObservations());
    settings->setAircraftDescription(forumInteractionForm->getAircraftDescription());
    Utils::PHPBB php("http://forum.taulabs.org", this);
    if (!php.login(forumInteractionForm->getUserName(), forumInteractionForm->getPassword())) {
       QMessageBox::warning(this, tr("Forum login"), tr("Forum login failed, probably wrong username or password"));
       forumInteractionForm->deleteLater();
       return;
    }

    QString message0 = tr(
                "[b]Aircraft description[/b]:%0\n\n\n"
                "[b]Observations[/b]:%1\n\n\n"
                "[b]Measured properties[/b]"
                "[table][tr][td][/td]"
                "[td][b]Gain[/b][/td]"
                "[td][b]Bias[/b][/td]"
                "[td][b]Tau(s)[/b][/td]"
                "[td][b]Noise[/b][/td][/tr]"
                "[tr][td][b]Roll[/b][/td]"
                "[td]%2[/td]"
                "[td]%3[/td]"
                "[td]%4[/td]"
                "[td]%5[/td][/tr]"
                "[tr][td][b]Pitch[/b][/td]"
                "[td]%6[/td]"
                "[td]%7[/td]"
                "[td]%8[/td]"
                "[td]%9[/td][/tr][/table]"
                "[b]\n\nTuning aggressiveness [/b]"
                "[table][tr][td][b]Damping[/b][/td]"
                "[td]%10[/td][/tr]"
                "[tr][td][b]Noise sensitivity[/b][/td]"
                "[td]%11[/td][/tr]"
                "[tr][td][b]Natural frequency[/b][/td]"
                "[td]%12[/td][/tr][/table]")
            .arg(forumInteractionForm->getAircraftDescription()).arg(forumInteractionForm->getObservations())
            .arg(m_autotune->measuredRollGain->text()).arg(m_autotune->measuredRollBias->text())
            .arg(m_autotune->rollTau->text()).arg(m_autotune->measuredRollNoise->text())
            .arg(m_autotune->measuredPitchGain->text()).arg(m_autotune->measuredPitchBias->text())
            .arg(m_autotune->pitchTau->text()).arg(m_autotune->measuredPitchNoise->text())
            .arg(m_autotune->lblDamp->text()).arg(m_autotune->lblNoise->text())
            .arg(m_autotune->wn->text());
    QString message1 = tr(
                "[b]\n\nComputed Values[/b]"
                "[table][tr][td][/td]"
                "[td][b]RateKp[/b][/td]"
                "[td][b]RateKi[/b][/td]"
                "[td][b]RateKd[/b][/td][/tr]"
                "[tr][td][b]Roll[/b][/td]"
                "[td]%1[/td]"
                "[td]%2[/td]"
                "[td]%3[/td][/tr]"
                "[tr][td][b]Pitch[/b][/td]"
                "[td]%4[/td]"
                "[td]%5[/td]"
                "[td]%6[/td][/tr]"
                "[tr][td][b]Outer Kp[/b][/td]"
                "[td]%7[/td]"
                "[td]-[/td]"
                "[td]-[/td][/tr]"
                "[tr][td][b]Derivative cutoff[/b][/td]"
                "[td]%8[/td]"
                "[td]-[/td]"
                "[td]-[/td][/tr][/table]"
                "\n\n")
            .arg(m_autotune->rollRateKp->text()).arg(m_autotune->rollRateKi->text()).arg(m_autotune->rollRateKd->text())
            .arg(m_autotune->pitchRateKp->text()).arg(m_autotune->pitchRateKi->text()).arg(m_autotune->pitchRateKd->text())
            .arg(m_autotune->lblOuterKp->text()).arg(m_autotune->derivativeCutoff->text());

    QString message = message0 + message1;
    if(php.postReply(FORUM_SHARING_FORUM, FORUM_SHARING_THREAD, "Autotune Results", message)) {
        QMessageBox::information(this, tr("Autotune results sharing"), tr("Thank you for sharing your results"));
    } else {
        QMessageBox::warning(this, tr("Autotune results sharing"), tr("Ooops, something went wrong, your results were not shared"));
    }
    forumInteractionForm->deleteLater();
}

/**
 * @brief ConfigAutotuneWidget::approveSettings
 * @param data The system ident values
 * @return True if these are ok, False if not
 *
 * For values that seem potentially problematic
 * a dialog will explicitly check the user wants
 * to apply them.
 */
bool ConfigAutotuneWidget::approveSettings(
        SystemIdent::DataFields systemIdentData)
{
    // Check the axis gains
    if (systemIdentData.Beta[SystemIdent::BETA_ROLL] < 6 ||
        systemIdentData.Beta[SystemIdent::BETA_PITCH] < 6) {

        int ans = QMessageBox::warning(this,tr("Extreme values"),
                                     tr("Your roll or pitch gain was lower than expected. This will result in large PID values. "
                                                           "Do you still want to proceed?"), QMessageBox::Yes,QMessageBox::No);
        if (ans == QMessageBox::No)
            return false;
    }

    // Check the response speed
    if (exp(systemIdentData.Tau) > 0.1) {

        int ans = QMessageBox::warning(this,tr("Extreme values"),
                                     tr("Your estimated response speed (tau) is slower than normal. This will result in large PID values. "
                                                           "Do you still want to proceed?"), QMessageBox::Yes,QMessageBox::No);
        if (ans == QMessageBox::No)
            return false;
    }

    return true;
}

/**
  * Called whenever the gain ratios or measured values
  * are changed
  */
void ConfigAutotuneWidget::recomputeStabilization()
{
    SystemIdent *systemIdent = SystemIdent::GetInstance(getObjectManager());
    Q_ASSERT(systemIdent);
    if(!systemIdent)
        return;

    StabilizationSettings *stabilizationSettings = StabilizationSettings::GetInstance(getObjectManager());
    Q_ASSERT(stabilizationSettings);
    if(!stabilizationSettings)
        return;

    SystemIdent::DataFields systemIdentData = systemIdent->getData();
    stabSettings = stabilizationSettings->getData();

    // These three parameters define the desired response properties
    // - rate scale in the fraction of the natural speed of the system
    //   to strive for.
    // - damp is the amount of damping in the system. higher values
    //   make oscillations less likely
    // - ghf is the amount of high frequency gain and limits the influence
    //   of noise
    const double ghf = m_autotune->rateNoise->value() / 1000.0;
    const double damp = m_autotune->rateDamp->value() / 100.0;

    double tau = exp(systemIdentData.Tau);
    double beta_roll = systemIdentData.Beta[SystemIdent::BETA_ROLL];
    double beta_pitch = systemIdentData.Beta[SystemIdent::BETA_PITCH];

    double wn = 1/tau;
    double tau_d = 0;
    for (int i = 0; i < 30; i++) {
        double tau_d_roll = (2*damp*tau*wn - 1)/(4*tau*damp*damp*wn*wn - 2*damp*wn - tau*wn*wn + exp(beta_roll)*ghf);
        double tau_d_pitch = (2*damp*tau*wn - 1)/(4*tau*damp*damp*wn*wn - 2*damp*wn - tau*wn*wn + exp(beta_pitch)*ghf);

        // Select the slowest filter property
        tau_d = (tau_d_roll > tau_d_pitch) ? tau_d_roll : tau_d_pitch;
        wn = (tau + tau_d) / (tau*tau_d) / (2 * damp + 2);
    }

    // Set the real pole position. The first pole is quite slow, which
    // prevents the integral being too snappy and driving too much
    // overshoot.
    const double a = ((tau+tau_d) / tau / tau_d - 2 * damp * wn) / 20.0;
    const double b = ((tau+tau_d) / tau / tau_d - 2 * damp * wn - a);

    qDebug() << "ghf: " << ghf;
    qDebug() << "wn: " << wn << "tau_d: " << tau_d;
    qDebug() << "a: " << a << " b: " << b;

    // Calculate the gain for the outer loop by approximating the
    // inner loop as a single order lpf. Set the outer loop to be
    // critically damped;
    const double zeta_o = 1.3;
    const double kp_o = 1 / 4.0 / (zeta_o * zeta_o) / (1/wn);

    // For now just run over roll and pitch
    for (int i = 0; i < 2; i++) {
        double beta = exp(systemIdentData.Beta[i]);

        double ki = a * b * wn * wn * tau * tau_d / beta;
        double kp = tau * tau_d * ((a+b)*wn*wn + 2*a*b*damp*wn) / beta - ki*tau_d;
        double kd = (tau * tau_d * (a*b + wn*wn + (a+b)*2*damp*wn) - 1) / beta - kp * tau_d;

        switch(i) {
        case 0: // Roll
            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KP] = kp;
            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KI] = ki;
            stabSettings.RollRatePID[StabilizationSettings::ROLLRATEPID_KD] = kd;
            stabSettings.RollPI[StabilizationSettings::ROLLPI_KP] = kp_o;
            stabSettings.RollPI[StabilizationSettings::ROLLPI_KI] = 0;
            break;
        case 1: // Pitch
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KP] = kp;
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KI] = ki;
            stabSettings.PitchRatePID[StabilizationSettings::PITCHRATEPID_KD] = kd;
            stabSettings.PitchPI[StabilizationSettings::PITCHPI_KP] = kp_o;
            stabSettings.PitchPI[StabilizationSettings::PITCHPI_KI] = 0;
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
    m_autotune->lblOuterKp->setText(QString::number(stabSettings.RollPI[StabilizationSettings::ROLLPI_KP]));

    m_autotune->derivativeCutoff->setText(QString::number(stabSettings.DerivativeCutoff));
    m_autotune->rollTau->setText(QString::number(tau,'g',3));
    m_autotune->pitchTau->setText(QString::number(tau,'g',3));
    m_autotune->wn->setText(QString::number(wn / 2 / M_PI, 'f', 1));
    m_autotune->lblDamp->setText(QString::number(damp, 'g', 2));
    m_autotune->lblNoise->setText(QString::number(ghf * 100, 'g', 2) + " %");

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
